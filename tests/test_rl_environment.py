"""Tests for the RL environment and reward function."""

from __future__ import annotations

import numpy as np
import pytest

from irrigation.actuators.base import IrrigationCommand
from irrigation.actuators.simulation import SimulatedActuator
from irrigation.crops.chili import ChiliProfile
from irrigation.rl.environment import IrrigationEnvironment, IrrigationState
from irrigation.rl.reward import RewardFunction
from irrigation.sensors.simulation import (
    CombinedSimulatedSensor,
    SimulatedSoilMoistureSensor,
    SimulatedWeatherSensor,
)
from irrigation.zone_config import ZoneConfig


def _make_env(initial_moisture: float = 50.0) -> IrrigationEnvironment:
    soil = SimulatedSoilMoistureSensor(initial_moisture_pct=initial_moisture, seed=42)
    weather = SimulatedWeatherSensor(seed=42)
    sensor = CombinedSimulatedSensor(soil, weather)
    actuator = SimulatedActuator(soil_sensor=soil)
    return IrrigationEnvironment(
        sensor=sensor,
        actuator=actuator,
        crop=ChiliProfile(),
    )


class TestIrrigationState:
    def test_to_observation_shape(self):
        state = IrrigationState(
            soil_moisture_pct=60.0,
            temperature_celsius=28.0,
            humidity_pct=70.0,
            hour=8.0,
            is_raining=False,
            growth_stage=2,
            current_day=65,
        )
        obs = state.to_observation(growing_season_days=150)
        assert obs.shape == (7,)
        assert obs.dtype == np.float32

    def test_to_observation_clipped(self):
        state = IrrigationState(
            soil_moisture_pct=105.0,   # above 100
            temperature_celsius=50.0,  # above 40°C max
            humidity_pct=70.0,
            hour=8.0,
            is_raining=False,
            growth_stage=2,
            current_day=65,
        )
        obs = state.to_observation()
        assert float(obs[0]) <= 1.0
        assert float(obs[1]) <= 1.0

    def test_to_observation_values(self):
        state = IrrigationState(
            soil_moisture_pct=60.0,
            temperature_celsius=28.0,
            humidity_pct=70.0,
            hour=6.0,
            is_raining=True,
            growth_stage=2,
            current_day=75,
        )
        obs = state.to_observation(growing_season_days=150)
        assert obs[0] == pytest.approx(0.6)   # 60/100
        assert obs[4] == pytest.approx(1.0)   # is_raining=True
        assert obs[5] == pytest.approx(0.5)   # stage 2/4


class TestIrrigationEnvironment:
    def test_observe_returns_state(self):
        env = _make_env()
        state = env.observe()
        assert isinstance(state, IrrigationState)

    def test_observe_moisture_in_range(self):
        env = _make_env(initial_moisture=60.0)
        state = env.observe()
        assert 0.0 <= state.soil_moisture_pct <= 100.0

    def test_step_returns_tuple(self):
        env = _make_env()
        env.observe()
        next_state, reward, done = env.step(water_litres=0.0)
        assert isinstance(next_state, IrrigationState)
        assert isinstance(reward, float)
        assert done is False

    def test_step_irrigation_increases_moisture(self):
        env = _make_env(initial_moisture=30.0)
        env.observe()
        state_before = env.observe()
        env.step(water_litres=10.0)
        state_after = env.observe()
        assert state_after.soil_moisture_pct >= state_before.soil_moisture_pct

    def test_observation_size(self):
        env = _make_env()
        assert env.observation_size == 10

    def test_sim_day_override(self):
        env = _make_env()
        env._sim_day = 65
        state = env.observe()
        assert state.current_day == 65
        assert state.growth_stage == 2  # flowering stage


class TestRewardFunction:
    def setup_method(self):
        self.crop = ChiliProfile()
        self.zone = ZoneConfig()
        self.reward_fn = RewardFunction(self.crop, self.zone)

    def test_optimal_moisture_no_water_positive_reward(self):
        t = self.crop.moisture_thresholds
        mid = (t.optimal_min + t.optimal_max) / 2
        cmd = IrrigationCommand(water_litres=0.0)
        reward = self.reward_fn.compute(soil_moisture_pct=mid, command=cmd)
        assert reward > 0.0

    def test_drought_stress_penalises_reward(self):
        t = self.crop.moisture_thresholds
        cmd = IrrigationCommand(water_litres=0.0)
        reward_stressed = self.reward_fn.compute(
            soil_moisture_pct=t.wilting_point, command=cmd
        )
        reward_optimal = self.reward_fn.compute(
            soil_moisture_pct=t.optimal_min, command=cmd
        )
        assert reward_stressed < reward_optimal

    def test_more_water_reduces_reward_at_optimal_moisture(self):
        t = self.crop.moisture_thresholds
        mid = (t.optimal_min + t.optimal_max) / 2
        reward_none = self.reward_fn.compute(
            soil_moisture_pct=mid, command=IrrigationCommand(water_litres=0.0)
        )
        reward_irrigate = self.reward_fn.compute(
            soil_moisture_pct=mid, command=IrrigationCommand(water_litres=8.0)
        )
        assert reward_none > reward_irrigate

    def test_rain_bonus_for_skipping_irrigation(self):
        t = self.crop.moisture_thresholds
        mid = (t.optimal_min + t.optimal_max) / 2
        cmd = IrrigationCommand(water_litres=0.0)
        reward_no_rain = self.reward_fn.compute(
            soil_moisture_pct=mid, command=cmd, is_raining=False
        )
        reward_rain = self.reward_fn.compute(
            soil_moisture_pct=mid, command=cmd, is_raining=True
        )
        assert reward_rain > reward_no_rain

    def test_overwatering_penalised(self):
        t = self.crop.moisture_thresholds
        cmd = IrrigationCommand(water_litres=0.0)
        reward_optimal = self.reward_fn.compute(
            soil_moisture_pct=t.optimal_max, command=cmd
        )
        reward_overwater = self.reward_fn.compute(
            soil_moisture_pct=t.field_capacity + 5.0, command=cmd
        )
        assert reward_optimal > reward_overwater
