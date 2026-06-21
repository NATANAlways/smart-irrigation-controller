"""Simulated sensor implementations for testing and development.

These classes reproduce realistic sensor behaviour without requiring physical
hardware, making it possible to train the RL agent at full speed on any machine.

All sensors are step-based: each call to read() advances the simulation by a
fixed step_hours interval (default 1.0 h = 1 hour) rather than using the
real wall clock. This matches the NASA POWER hourly data exactly and allows
training to run thousands of steps per second while simulating realistic dynamics.

Typical usage:
    soil = SimulatedSoilMoistureSensor(initial_moisture_pct=60.0)
    weather = SimulatedWeatherSensor()
    sensor = CombinedSimulatedSensor(soil, weather)
    env = IrrigationEnvironment(sensor, actuator, crop)
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

from irrigation.sensors.base import SensorInterface, SensorReading


class SimulatedSoilMoistureSensor:
    """Soil moisture sensor backed by a step-based water-balance simulation.

    Each call to read() advances the simulation by exactly step_hours,
    applying evapotranspiration independently of real elapsed time.

    Args:
        initial_moisture_pct: Starting soil moisture (0–100 %).
        et_rate_per_hour: Evapotranspiration rate in % per simulated hour.
            Default 0.083 = 2%/day which matches Jaffna ET₀ ≈ 6mm/day for 3m² zone:
            ET loss/day = (6mm × 3m²) / 900L soil × 100 = 2%/day → 0.083%/hr
        step_hours: Simulated time per step in hours (default 1.0 = 1 hour).
        initial_hour: Starting hour of day for the simulated clock (0–24).
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        initial_moisture_pct: float = 50.0,
        et_rate_per_hour: float = 0.083,
        step_hours: float = 1.0,
        initial_hour: float = 6.0,
        seed: int | None = None,
    ) -> None:
        self._moisture = initial_moisture_pct
        self.et_rate_per_hour = et_rate_per_hour
        self.step_hours = step_hours
        self._simulated_hour = initial_hour
        if seed is not None:
            random.seed(seed)

    @property
    def moisture_pct(self) -> float:
        return self._moisture

    def irrigate(self, amount_pct: float) -> None:
        """Add soil moisture as if irrigation were applied.

        Args:
            amount_pct: Moisture increase in percentage points.
        """
        self._moisture = min(100.0, self._moisture + amount_pct)

    def apply_rain(self, rainfall_mm: float, cap_pct: float = 100.0) -> None:
        """Increase moisture as if rain fell (1 mm ≈ 0.5 % moisture).

        Args:
            rainfall_mm: Rainfall in millimetres.
            cap_pct: Maximum moisture after rain. Pass field_capacity so that
                rainfall drains away above saturation (physically correct for
                a raised bed). Defaults to 100.0 to preserve legacy behaviour
                when cap is not specified.
        """
        self._moisture = min(cap_pct, self._moisture + rainfall_mm * 0.5)

    def reset(self, initial_moisture_pct: float = 50.0, initial_hour: float = 6.0) -> None:
        """Reset moisture and simulated clock for a new training episode."""
        self._moisture = initial_moisture_pct
        self._simulated_hour = initial_hour

    def read(self) -> SensorReading:
        # Apply ET for exactly one step.
        et_loss = self.et_rate_per_hour * self.step_hours
        self._moisture = max(0.0, self._moisture - et_loss)

        noise = random.gauss(0, 0.3)
        noisy_moisture = max(0.0, min(100.0, self._moisture + noise))

        # Build a simulated timestamp from the internal hour counter.
        base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        simulated_time = base + timedelta(hours=self._simulated_hour % 24)

        self._simulated_hour += self.step_hours

        return SensorReading(
            timestamp=simulated_time,
            soil_moisture_pct=round(noisy_moisture, 1),
        )


class SimulatedWeatherSensor:
    """Weather sensor with a step-based simulated clock.

    Temperature follows a realistic diurnal cycle keyed to the simulated hour,
    not the real wall clock. Rain is generated stochastically each step.

    Args:
        base_temp_celsius: Mean daily temperature.
        temp_amplitude: Half-range of the diurnal temperature swing.
        base_humidity_pct: Mean relative humidity.
        is_rainy_season: Whether to simulate more frequent rainfall.
        step_hours: Simulated time per step in hours (default 1.0 = 1 hour).
        initial_hour: Starting hour of day for the simulated clock (0–24).
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        base_temp_celsius: float = 28.0,
        temp_amplitude: float = 5.0,
        base_humidity_pct: float = 70.0,
        is_rainy_season: bool = False,
        step_hours: float = 1.0,
        initial_hour: float = 6.0,
        seed: int | None = None,
    ) -> None:
        self.base_temp = base_temp_celsius
        self.temp_amplitude = temp_amplitude
        self.base_humidity = base_humidity_pct
        self.is_rainy_season = is_rainy_season
        self.step_hours = step_hours
        self._simulated_hour = initial_hour
        if seed is not None:
            random.seed(seed)
        self._rain_active: bool = False

    def _temperature_at(self, hour: float) -> float:
        """Return estimated temperature for the given simulated hour (0–24)."""
        # Peak temperature around 14:00.
        radians = 2 * math.pi * (hour - 14) / 24
        return self.base_temp + self.temp_amplitude * math.cos(radians)

    def set_rain(self, raining: bool) -> None:
        """Manually override rain state (useful in controlled tests)."""
        self._rain_active = raining

    def reset(self, initial_hour: float = 6.0) -> None:
        """Reset the simulated clock for a new training episode."""
        self._simulated_hour = initial_hour
        self._rain_active = False

    def read(self) -> SensorReading:
        hour = self._simulated_hour % 24

        temp = self._temperature_at(hour) + random.gauss(0, 0.5)
        humidity = self.base_humidity + random.gauss(0, 3.0)
        humidity = max(0.0, min(100.0, humidity))

        rain_prob = 0.15 if self.is_rainy_season else 0.03
        if self._rain_active or random.random() < rain_prob:
            is_raining = True
            rainfall_mm = random.uniform(0.5, 5.0)
        else:
            is_raining = False
            rainfall_mm = 0.0

        self._simulated_hour += self.step_hours

        return SensorReading(
            temperature_celsius=round(temp, 1),
            humidity_pct=round(humidity, 1),
            is_raining=is_raining,
            rainfall_mm=round(rainfall_mm, 2),
        )


class CombinedSimulatedSensor(SensorInterface):
    """Merges soil moisture and weather sensors into a single SensorReading.

    This is the sensor to pass to IrrigationEnvironment during simulation
    and training. Both underlying sensors share the same step_hours so that
    the simulated clocks stay in sync.

    Args:
        soil_sensor: A SimulatedSoilMoistureSensor instance.
        weather_sensor: A SimulatedWeatherSensor instance.
    """

    def __init__(
        self,
        soil_sensor: SimulatedSoilMoistureSensor,
        weather_sensor: SimulatedWeatherSensor,
    ) -> None:
        self.soil_sensor = soil_sensor
        self.weather_sensor = weather_sensor

    def read(self) -> SensorReading:
        """Read both sensors and merge into one SensorReading."""
        soil = self.soil_sensor.read()
        weather = self.weather_sensor.read()

        # Apply any rain to soil moisture.
        if weather.is_raining and weather.rainfall_mm > 0:
            self.soil_sensor.apply_rain(weather.rainfall_mm)

        return SensorReading(
            timestamp=soil.timestamp,
            soil_moisture_pct=soil.soil_moisture_pct,
            temperature_celsius=weather.temperature_celsius,
            humidity_pct=weather.humidity_pct,
            is_raining=weather.is_raining,
            rainfall_mm=weather.rainfall_mm,
        )

    def reset(self, initial_moisture_pct: float = 50.0, initial_hour: float = 6.0) -> None:
        """Reset both sensors for a new training episode."""
        self.soil_sensor.reset(initial_moisture_pct, initial_hour)
        self.weather_sensor.reset(initial_hour)
