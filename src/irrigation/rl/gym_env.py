"""Gymnasium wrapper around IrrigationEnvironment for PPO training.

One episode = one full growing season. The simulated clock advances by
step_hours on every step, independent of real wall-clock time.

Step size: 1 hour (matches NASA POWER hourly data exactly)
Episode length: 150 days × 24 steps/day = 3,600 steps per episode

Curriculum Learning Phases:
    Phase 1 — Dry months only (Feb–Apr, Jaffna dry season)
              Agent learns basic irrigation skill in predictable conditions.
    Phase 2 — All months (dry + moderate + Northeast monsoon)
              Agent generalises to all real Jaffna weather patterns.

Weather source: Real NASA POWER hourly data (2004–2024, Uduvil, Jaffna).
Each step samples a real historical record matching the current hour of day.

Action space: Box(0, 1, shape=(1,)) — agent outputs a fraction in [0, 1]
              mapped to [0, zone.max_litres_per_event] litres.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from irrigation.actuators.simulation import SimulatedActuator
from irrigation.crops.base import CropProfile
from irrigation.crops.chili import ChiliProfile
from irrigation.rl.dynamic_stage import DynamicStageTracker
from irrigation.rl.environment import IrrigationEnvironment
from irrigation.rl.health_tracker import PlantHealthTracker
from irrigation.rl.vitality import VITALITY_CONFIGS, PlantVitalityTracker
from irrigation.sensors.historical import HistoricalWeatherSensor
from irrigation.sensors.simulation import SimulatedSoilMoistureSensor
from irrigation.weather.weather_data import WeatherDataLoader
from irrigation.zone_config import ZoneConfig

# Default path to the NASA POWER hourly CSV
_DEFAULT_CSV = (
    Path(__file__).parent.parent.parent.parent
    / "data" / "weather" / "uduvil_per_hour" / "uduvil_hourly_2004_2024.csv"
)


class IrrigationGymEnv(gymnasium.Env):
    """PPO-ready Gymnasium environment for irrigation control.

    Uses real NASA POWER hourly weather data for Uduvil, Jaffna.
    Real ET₀ drives soil moisture drain rate each step.

    Args:
        crop:            Crop profile. Defaults to ChiliProfile.
        zone:            Zone configuration. Defaults to ZoneConfig (3m² drip).
        step_hours:      Simulated time per step (default 1.0 = 1 hour).
        training_phase:  Curriculum phase (1=dry months, 2=all months).
        weather_csv:     Path to NASA POWER hourly CSV. Defaults to project data.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        crop: CropProfile | None = None,
        zone: ZoneConfig | None = None,
        step_hours: float = 1.0,
        training_phase: int = 1,
        weather_csv: str | Path | None = None,
        death_warmup_steps: int = 0,
    ) -> None:
        super().__init__()

        self.crop           = crop or ChiliProfile()
        self.zone           = zone or ZoneConfig()
        self.step_hours     = step_hours
        self.training_phase = training_phase
        self._steps_per_day = int(24 / step_hours)
        self._max_steps     = self.crop.growing_season_days * self._steps_per_day

        # Death-disabled warm-up — see config.yaml training.death_warmup_steps.
        # Counts total steps taken by this env instance (not reset per episode).
        self.death_warmup_steps  = death_warmup_steps
        self._total_steps        = 0
        self._last_is_raining: bool = False  # rain delay guard state

        # Continuous action: fraction of max irrigation volume.
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # 10 normalized continuous observation values, all in [0, 1].
        # [moisture, temp, humidity, hour, is_raining, stage, day, health, vitality, waterlog_risk]
        # waterlog_risk = consecutive_wet_hours / wet_hours_limit (0=safe, 1=dead)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )

        # Load real NASA POWER weather data
        csv_path = Path(weather_csv) if weather_csv else _DEFAULT_CSV
        self._loader = WeatherDataLoader(csv_path)

        # Soil sensor — moisture state is maintained here
        self._soil_sensor = SimulatedSoilMoistureSensor(step_hours=step_hours)

        # Historical weather sensor — replaces SimulatedWeatherSensor
        # Feeds real ET₀ to soil sensor each step
        # Use germination field capacity (most restrictive stage) as the
        # drainage cap for rain — soil can't exceed field capacity in a
        # well-drained raised bed regardless of rainfall intensity.
        _field_cap = self.crop.moisture_thresholds_for_stage(0).field_capacity
        self._sensor = HistoricalWeatherSensor(
            loader=self._loader,
            soil_sensor=self._soil_sensor,
            zone=self.zone,
            training_phase=training_phase,
            episode_hours=self._max_steps,
            field_capacity_pct=_field_cap,
        )

        self._actuator = SimulatedActuator(
            soil_sensor=self._soil_sensor,
            zone=self.zone,
        )
        self._env = IrrigationEnvironment(
            sensor=self._sensor,
            actuator=self._actuator,
            crop=self.crop,
            planting_date=date.today(),
            zone=self.zone,
        )

        self._step             = 0
        self._health_tracker   = PlantHealthTracker(self.crop)
        self._stage_tracker    = DynamicStageTracker(self.crop)
        self._vitality_tracker = PlantVitalityTracker(self.crop)

    def _make_obs(self, state) -> np.ndarray:
        """Build (10,) observation: 7 env values + health_score + vitality + waterlog_risk."""
        base_obs = np.clip(
            state.to_observation(self.crop.growing_season_days), 0.0, 1.0
        )
        current_stage = self._stage_tracker.current_stage
        wet_limit = VITALITY_CONFIGS[current_stage].wet_hours_limit
        wet_risk = float(self._vitality_tracker.consecutive_wet_hours) / max(1, wet_limit)
        extras = np.array(
            [self._health_tracker.health_score,
             self._vitality_tracker.vitality,
             min(1.0, wet_risk)],
            dtype=np.float32,
        )
        return np.concatenate([base_obs, extras])

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Update curriculum phase in the sensor (controls month filter)
        self._sensor.set_training_phase(self.training_phase)

        # Randomize initial soil moisture for diverse training scenarios.
        # Narrowed from (40, 80) — that range placed many episodes right at a
        # Stage-0 death boundary before the policy had learned anything.
        initial_moisture = float(self.np_random.uniform(50.0, 65.0))
        self._sensor.reset(initial_moisture_pct=initial_moisture, initial_hour=6.0)
        self._actuator.reset()

        self._step            = 0
        self._last_is_raining = False
        self._env._sim_day    = 0
        self._env._sim_stage  = 0
        self._env._last_reading = None
        self._health_tracker.reset()
        self._stage_tracker.reset()
        self._vitality_tracker.reset()

        state = self._env.observe()
        obs   = self._make_obs(state)
        return obs, {"phase": self.training_phase}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Sync simulated day and dynamic stage before the environment observes.
        self._env._sim_day   = self._step // self._steps_per_day
        self._env._sim_stage = self._stage_tracker.current_stage

        # Map agent output [0, 1] → [0, max_litres_per_event].
        water_fraction = float(np.clip(action[0], 0.0, 1.0))
        water_litres   = water_fraction * self.zone.max_litres_per_event

        # Rain delay guard: if it was raining last step and soil is already above
        # optimal moisture, force irrigation to zero. This mirrors the rain sensor
        # found in real irrigation controllers. A Gaussian PPO policy cannot
        # reliably output exactly 0.0, so spurious micro-irrigation during
        # multi-day monsoon rain would otherwise accumulate above field capacity.
        stage_for_guard = self._stage_tracker.current_stage
        t_guard = self.crop.moisture_thresholds_for_stage(stage_for_guard)
        if self._last_is_raining and self._soil_sensor.moisture_pct > t_guard.optimal_max:
            water_litres = 0.0

        next_state, reward, _ = self._env.step(water_litres)
        self._step += 1

        current_stage = self._stage_tracker.current_stage

        # Update dynamic stage tracker — may advance stage this step.
        stage_advanced = self._stage_tracker.update(
            moisture_pct=next_state.soil_moisture_pct,
        )

        # Update vitality — check plant death using current dynamic stage.
        self._vitality_tracker.update(
            moisture_pct=next_state.soil_moisture_pct,
            stage=current_stage,
        )

        # Update health tracker with dynamic stage.
        self._health_tracker.update(
            moisture_pct=next_state.soil_moisture_pct,
            stage=current_stage,
            water_litres=water_litres,
        )

        # Apply death penalty if plant died. During the death-disabled
        # warm-up, the episode continues (vitality resets) so the agent can
        # keep experiencing later growth stages while still close to random.
        self._total_steps += 1
        plant_dead   = self._vitality_tracker.is_dead
        death_active = self._total_steps > self.death_warmup_steps
        if plant_dead:
            reward += self._vitality_tracker.death_penalty
            if not death_active:
                self._vitality_tracker.reset()

        obs        = self._make_obs(next_state)
        terminated = self._step >= self._max_steps or (plant_dead and death_active)

        info = {
            "day":            next_state.current_day,
            "calendar_stage": next_state.growth_stage,
            "dynamic_stage":  current_stage,
            "stage_name":     self._stage_tracker.stage_name,
            "days_in_stage":  self._stage_tracker.days_in_stage,
            "stage_advanced": stage_advanced,
            "health_ratio":   self._stage_tracker.health_ratio,
            "moisture":       next_state.soil_moisture_pct,
            "water_litres":   water_litres,
            "health_score":   self._health_tracker.health_score,
            "stress_ratio":   self._health_tracker.stress_ratio,
            "vitality":       self._vitality_tracker.vitality,
            "phase":          self.training_phase,
        }

        # Update rain delay state for next step.
        self._last_is_raining = next_state.is_raining

        # At episode end add full summaries from all trackers.
        if terminated:
            info.update(self._health_tracker.summary())
            info.update(self._stage_tracker.summary())
            info.update(self._vitality_tracker.death_info())

        return obs, float(reward), terminated, False, info
