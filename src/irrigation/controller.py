"""Main irrigation controller orchestrating sensors, actuators and RL agent."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

from irrigation.actuators.base import ActuatorInterface
from irrigation.config import ControllerConfig
from irrigation.crops import get_crop_profile
from irrigation.rl.agent import QLearningAgent
from irrigation.rl.environment import IrrigationEnvironment
from irrigation.sensors.base import SensorInterface
from irrigation.zone_config import ZoneConfig

logger = logging.getLogger(__name__)


def _build_components(
    config: ControllerConfig,
) -> tuple[SensorInterface, ActuatorInterface]:
    """Instantiate sensor and actuator based on configuration.

    In simulation mode (or when hardware libraries are unavailable) simulated
    components are returned so the controller can run without a Raspberry Pi.
    """
    if config.simulation_mode:
        from irrigation.sensors.simulation import (
            SimulatedSoilMoistureSensor,
            SimulatedWeatherSensor,
        )
        from irrigation.actuators.simulation import SimulatedActuator
        from irrigation.sensors.base import SensorReading

        class _CombinedSimSensor(SensorInterface):
            """Composite sensor that merges soil and weather readings."""

            def __init__(self) -> None:
                self._soil = SimulatedSoilMoistureSensor()
                self._weather = SimulatedWeatherSensor(
                    base_temp_celsius=config.climate.avg_daily_temp_celsius,
                    is_rainy_season=datetime.now().month
                    in config.climate.rainy_season_months,
                )
                # Expose soil sensor so the actuator can update moisture.
                self.soil = self._soil

            def read(self) -> SensorReading:
                soil = self._soil.read()
                weather = self._weather.read()
                return SensorReading(
                    timestamp=soil.timestamp,
                    soil_moisture_pct=soil.soil_moisture_pct,
                    temperature_celsius=weather.temperature_celsius,
                    humidity_pct=weather.humidity_pct,
                    is_raining=weather.is_raining,
                    rainfall_mm=weather.rainfall_mm,
                )

        sensor = _CombinedSimSensor()
        actuator = SimulatedActuator(soil_sensor=sensor.soil, zone=ZoneConfig())
        return sensor, actuator

    # Real hardware path.
    from irrigation.sensors.soil import SoilMoistureSensor
    from irrigation.sensors.weather import WeatherSensor
    from irrigation.actuators.valve import ValveActuator
    from irrigation.sensors.base import SensorReading

    class _CombinedHWSensor(SensorInterface):
        """Merges hardware soil and weather readings into a single SensorReading."""

        def __init__(self) -> None:
            self._soil = SoilMoistureSensor(
                pin=config.sensor.soil_moisture_pin,
                adc_channel=config.sensor.adc_channel,
            )
            self._weather = WeatherSensor(
                dht_pin=config.sensor.dht_pin,
                rain_pin=config.sensor.rain_sensor_pin,
            )

        def read(self) -> SensorReading:
            soil = self._soil.read()
            weather = self._weather.read()
            return SensorReading(
                timestamp=soil.timestamp,
                soil_moisture_pct=soil.soil_moisture_pct,
                temperature_celsius=weather.temperature_celsius,
                humidity_pct=weather.humidity_pct,
                is_raining=weather.is_raining,
                rainfall_mm=weather.rainfall_mm,
            )

        def close(self) -> None:
            self._soil.close()
            self._weather.close()

    sensor = _CombinedHWSensor()
    actuator = ValveActuator(
        valve_pin=config.actuator.valve_pin,
        pump_pin=config.actuator.pump_pin,
    )
    return sensor, actuator


class IrrigationController:
    """Top-level controller that ties together all subsystems.

    Typical usage::

        config = ControllerConfig.default_sri_lanka()
        config.simulation_mode = True
        controller = IrrigationController(config)
        controller.run()

    Args:
        config: Controller configuration.
    """

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        self._sensor, self._actuator = _build_components(self.config)
        self._crop = get_crop_profile(self.config.crop_profile)
        self._env = IrrigationEnvironment(
            sensor=self._sensor,
            actuator=self._actuator,
            crop=self._crop,
            n_soil_bins=self.config.rl.n_soil_moisture_bins,
            n_temp_bins=self.config.rl.n_temperature_bins,
            n_time_bins=self.config.rl.n_time_bins,
        )
        self._agent = QLearningAgent(
            env=self._env,
            learning_rate=self.config.rl.learning_rate,
            discount_factor=self.config.rl.discount_factor,
            exploration_rate=self.config.rl.exploration_rate,
            exploration_min=self.config.rl.exploration_min,
            exploration_decay=self.config.rl.exploration_decay,
        )
        self._load_model_if_exists()

    def _model_path(self) -> Path:
        return Path(self.config.data_dir) / self.config.rl.model_path

    def _load_model_if_exists(self) -> None:
        path = self._model_path()
        if path.exists():
            try:
                self._agent.load(path)
                logger.info("Loaded RL model from %s", path)
            except Exception as exc:
                logger.warning("Could not load model (%s) – starting fresh.", exc)

    def train(self, n_episodes: int = 500, steps_per_episode: int = 48) -> list[float]:
        """Train the RL agent and persist the resulting model.

        Args:
            n_episodes: Training episodes.
            steps_per_episode: Steps per episode (1 step ≈ 30 min of real time).

        Returns:
            Episode reward history.
        """
        logger.info("Starting RL training for %d episodes …", n_episodes)
        rewards = self._agent.train(n_episodes=n_episodes, steps_per_episode=steps_per_episode)
        self._agent.save(self._model_path())
        return rewards

    def decide(self) -> None:
        """Take one sensor reading, choose an action, and execute it.

        This method is intended to be called on a schedule (e.g. every 30 min
        via cron or a timer loop).
        """
        state = self._env.observe()
        action = self._agent.choose_greedy_action(state)
        _, reward, _ = self._env.step(action)
        logger.info(
            "Decision: action=%s reward=%.4f moisture_bin=%d",
            action.name,
            reward,
            state.soil_moisture_bin,
        )

    def run(self, interval_seconds: int | None = None) -> None:
        """Run the controller in a continuous loop.

        Args:
            interval_seconds: Seconds between decisions.  Defaults to the
                ``read_interval_seconds`` value from the sensor config.
        """
        interval = interval_seconds or self.config.sensor.read_interval_seconds
        logger.info("Controller running (interval=%ds). Press Ctrl+C to stop.", interval)
        try:
            while True:
                self.decide()
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Shutdown requested.")
        finally:
            self.close()

    def close(self) -> None:
        """Release all hardware resources."""
        self._sensor.close()
        self._actuator.close()
