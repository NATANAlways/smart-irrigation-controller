"""RL environment for the irrigation controller.

The environment wraps the physical (or simulated) sensors and actuators and
exposes a Gym-like interface that the PPO agent interacts with.

Observation space (7 continuous values, normalized to [0, 1]):
    - Soil moisture      soil_moisture_pct / 100
    - Temperature        temperature_celsius / 40
    - Humidity           humidity_pct / 100
    - Time of day        hour / 24
    - Is raining         0 or 1
    - Growth stage       stage / 4
    - Current day        current_day / growing_season_days

Action space:
    Continuous float in [0, 1] mapped to [0, zone.max_litres_per_event].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np

from irrigation.actuators.base import ActuatorInterface, IrrigationCommand
from irrigation.crops.base import CropProfile
from irrigation.rl.reward import RewardFunction
from irrigation.sensors.base import SensorInterface, SensorReading
from irrigation.zone_config import ZoneConfig

logger = logging.getLogger(__name__)


@dataclass
class IrrigationState:
    """The continuous state observed by the RL agent.

    Attributes:
        soil_moisture_pct: Raw soil moisture percentage (0-100).
        temperature_celsius: Air temperature in degrees Celsius.
        humidity_pct: Relative humidity percentage (0-100).
        hour: Hour of day with fractional minutes (0-24).
        is_raining: Whether rain is currently detected.
        growth_stage: Current growth stage index (0-4).
        current_day: Days since planting (0 to growing_season_days).
    """

    soil_moisture_pct: float
    temperature_celsius: float
    humidity_pct: float
    hour: float
    is_raining: bool
    growth_stage: int
    current_day: int

    def to_observation(self, growing_season_days: int = 150) -> np.ndarray:
        """Return a normalized float32 array for the PPO policy network."""
        obs = np.array(
            [
                self.soil_moisture_pct / 100.0,
                self.temperature_celsius / 40.0,
                self.humidity_pct / 100.0,
                self.hour / 24.0,
                float(self.is_raining),
                self.growth_stage / 4.0,
                self.current_day / growing_season_days,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, 0.0, 1.0)


class IrrigationEnvironment:
    """Gym-inspired environment wrapping sensors, actuators and reward logic.

    Args:
        sensor: Sensor implementation (real or simulated).
        actuator: Actuator implementation (real or simulated).
        crop: Crop profile providing stage thresholds and season length.
        planting_date: The date the crop was planted. Defaults to today.
        zone: Zone configuration for water calculations. Defaults to ZoneConfig().
    """

    def __init__(
        self,
        sensor: SensorInterface,
        actuator: ActuatorInterface,
        crop: CropProfile,
        planting_date: date | None = None,
        zone: ZoneConfig | None = None,
    ) -> None:
        self.sensor = sensor
        self.actuator = actuator
        self.crop = crop
        self.planting_date = planting_date or date.today()
        self.zone = zone or ZoneConfig()
        self.reward_fn = RewardFunction(crop, self.zone)
        self._last_reading: SensorReading | None = None
        # Set by IrrigationGymEnv during training to override wall-clock day.
        self._sim_day: int | None = None
        # Set by IrrigationGymEnv to use dynamic stage instead of calendar stage.
        self._sim_stage: int | None = None

    def _current_day(self) -> int:
        """Days since planting, clamped to the season length."""
        if self._sim_day is not None:
            return self._sim_day
        days = (date.today() - self.planting_date).days
        return max(0, min(days, self.crop.growing_season_days))

    def _growth_stage(self, current_day: int) -> int:
        """Return growth stage — dynamic override if set, else calendar-based."""
        if self._sim_stage is not None:
            return self._sim_stage
        return self.crop.stage_for_day(current_day)

    def observe(self) -> IrrigationState:
        """Take a fresh sensor reading and return the current state."""
        reading = self.sensor.read()
        self._last_reading = reading
        current_day = self._current_day()
        return IrrigationState(
            soil_moisture_pct=reading.soil_moisture_pct,
            temperature_celsius=reading.temperature_celsius,
            humidity_pct=reading.humidity_pct,
            hour=reading.timestamp.hour + reading.timestamp.minute / 60.0,
            is_raining=reading.is_raining,
            growth_stage=self._growth_stage(current_day),
            current_day=current_day,
        )

    def step(
        self, water_litres: float
    ) -> tuple[IrrigationState, float, bool]:
        """Execute an irrigation command and return (next_state, reward, done).

        Args:
            water_litres: Volume of water to apply in litres (0.0 = no irrigation).

        Returns:
            A three-tuple of (next_state, reward, done). done is always
            False for this continuous-control environment.
        """
        command = IrrigationCommand(water_litres=max(0.0, water_litres))
        if self._last_reading is None:
            self.observe()
        assert self._last_reading is not None
        is_raining = self._last_reading.is_raining

        self.actuator.execute(command)

        next_state = self.observe()

        reward = self.reward_fn.compute(
            soil_moisture_pct=next_state.soil_moisture_pct,
            command=command,
            is_raining=is_raining,
            stage=next_state.growth_stage,
        )

        logger.debug(
            "water=%.2fL moisture=%.1f%% stage=%d reward=%.4f",
            water_litres,
            next_state.soil_moisture_pct,
            next_state.growth_stage,
            reward,
        )
        return next_state, reward, False

    @property
    def observation_size(self) -> int:
        """Number of values in the observation vector fed to the PPO network.
        7 env values + health_score + vitality + waterlog_risk appended by IrrigationGymEnv = 10.
        """
        return 10
