"""Historical weather sensor using real NASA POWER Jaffna data.

Each episode replays a real, contiguous slice of the 20-year hourly record —
not an independent random sample per hour — so weather has realistic
day-to-day persistence (gradual drying trends, multi-day rain events, real
diurnal cycles).

Curriculum phases control which calendar months an episode's start hour may
fall in (the rest of the episode just walks forward through real history from
that point):

    Phase 1 — Yala season (Jan/Feb/Mar start) — dry conditions
    Phase 2 — Maha season (Aug/Sep start) — Northeast monsoon conditions
    Phase 3 — Any start month — maximum variability

ET₀ → soil drain rate conversion:
    Moisture %/hr = ET₀_mm / (root_depth_m × 10)
"""

from __future__ import annotations

from irrigation.sensors.base import SensorInterface, SensorReading
from irrigation.sensors.simulation import SimulatedSoilMoistureSensor
from irrigation.weather.weather_data import MAHA_MONTHS, YALA_MONTHS, WeatherDataLoader
from irrigation.zone_config import ZoneConfig


class HistoricalWeatherSensor(SensorInterface):
    """Season-aware sensor backed by a continuous real-history replay.

    Args:
        loader:          WeatherDataLoader with the hourly CSV loaded.
        soil_sensor:     SimulatedSoilMoistureSensor to update each step.
        zone:            ZoneConfig for ET₀ → moisture conversion.
        training_phase:  1 = Yala start, 2 = Maha start, 3 = any start month.
        episode_hours:   Number of steps in one episode (used to pick a valid
                         start index that doesn't run past the end of history).
    """

    def __init__(
        self,
        loader: WeatherDataLoader,
        soil_sensor: SimulatedSoilMoistureSensor,
        zone: ZoneConfig,
        training_phase: int = 1,
        episode_hours: int = 3600,
        field_capacity_pct: float = 85.0,
    ) -> None:
        self.loader              = loader
        self.soil_sensor         = soil_sensor
        self.zone                = zone
        self.training_phase      = training_phase
        self.episode_hours       = episode_hours
        self._field_capacity_pct = field_capacity_pct

        self._index: int = 0

    def set_training_phase(self, phase: int) -> None:
        """Switch curriculum phase — takes effect on next reset()."""
        self.training_phase = phase

    def _allowed_start_months(self) -> list[int] | None:
        if self.training_phase == 1:
            return YALA_MONTHS
        if self.training_phase == 2:
            return MAHA_MONTHS
        return None

    def reset(
        self,
        initial_moisture_pct: float = 50.0,
        initial_hour: float = 6.0,
    ) -> None:
        """Reset for a new episode — picks a new random real-history start point."""
        self._index = self.loader.random_start_index(
            episode_hours=self.episode_hours,
            allowed_months=self._allowed_start_months(),
        )
        self.soil_sensor.reset(initial_moisture_pct, initial_hour)

    def read(self) -> SensorReading:
        """Advance one step through the real-history replay, update soil, return reading."""
        record = self.loader.record_at(self._index)

        # Convert ET₀ (mm/hr) → soil moisture drain rate (%/hr)
        et_rate = record.et0_mm / (self.zone.root_depth_m * 10)
        self.soil_sensor.et_rate_per_hour = et_rate

        if record.is_raining and record.rain_mm > 0:
            self.soil_sensor.apply_rain(record.rain_mm, cap_pct=self._field_capacity_pct)

        soil = self.soil_sensor.read()

        self._index += 1

        return SensorReading(
            timestamp=soil.timestamp,
            soil_moisture_pct=soil.soil_moisture_pct,
            temperature_celsius=record.temperature,
            humidity_pct=record.humidity_pct,
            is_raining=record.is_raining,
            rainfall_mm=record.rain_mm,
        )

    def close(self) -> None:
        pass
