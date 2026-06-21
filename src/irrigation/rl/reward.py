"""Reward function for the irrigation RL agent.

The reward signal balances three objectives:
1. Plant health   - keep soil moisture in the stage-specific optimal range.
2. Water savings  - penalise water use proportional to zone maximum.
3. Emergency care - strongly penalise ignoring critically dry soil.

A rain bonus is applied when irrigation is correctly withheld during rain.
"""

from __future__ import annotations

from irrigation.config_loader import load_config
from irrigation.crops.base import CropProfile
from irrigation.actuators.base import IrrigationCommand
from irrigation.zone_config import ZoneConfig

# Load reward weights from config.yaml → reward section
_r = load_config()["reward"]


class RewardFunction:
    """Compute scalar rewards for the RL agent.

    Default weights come from config.yaml → reward section.
    Override by passing explicit values (useful for experiments).

    Args:
        crop: The active crop profile providing stage-specific moisture thresholds.
        zone: Zone configuration used to normalise water penalties.
        water_penalty_weight: Penalty weight for water use.
        stress_penalty_weight: Penalty weight for plant stress.
        overwater_penalty: Penalty for exceeding field capacity.
        rain_bonus: Bonus for skipping irrigation when raining.
    """

    def __init__(
        self,
        crop: CropProfile,
        zone: ZoneConfig | None = None,
        water_penalty_weight: float = _r["water_penalty_weight"],
        stress_penalty_weight: float = _r["stress_penalty_weight"],
        overwater_penalty: float = _r["overwater_penalty"],
        rain_bonus: float = _r["rain_bonus"],
    ) -> None:
        self.crop = crop
        self.zone = zone or ZoneConfig()
        self.water_penalty_weight = water_penalty_weight
        self.stress_penalty_weight = stress_penalty_weight
        self.overwater_penalty = overwater_penalty
        self.rain_bonus = rain_bonus

    def compute(
        self,
        soil_moisture_pct: float,
        command: IrrigationCommand,
        is_raining: bool = False,
        stage: int = 0,
    ) -> float:
        """Compute the reward for a single time-step.

        Args:
            soil_moisture_pct: Soil moisture after executing the command.
            command: The irrigation command that was executed.
            is_raining: Whether it was raining when the command was issued.
            stage: Current growth stage index (0-4).

        Returns:
            A scalar reward value.
        """
        reward = 0.0
        t = self.crop.moisture_thresholds_for_stage(stage)

        # Plant health: penalise stress using stage-specific thresholds.
        stress = self.crop.stress_level_for_stage(soil_moisture_pct, stage)
        reward -= self.stress_penalty_weight * stress

        # Penalise over-watering above field capacity.
        if soil_moisture_pct > t.field_capacity:
            excess = soil_moisture_pct - t.field_capacity
            reward -= self.overwater_penalty * (excess / 10.0)

        # Reward for maintaining optimal moisture for the current stage.
        if t.optimal_min <= soil_moisture_pct <= t.optimal_max:
            reward += 0.5

        # Water conservation: penalise proportional to fraction of zone maximum.
        # Penalty stays in [0, water_penalty_weight] regardless of zone size.
        water_fraction = command.water_litres / self.zone.max_litres_per_event
        reward -= self.water_penalty_weight * water_fraction

        # Bonus for not irrigating when it's raining.
        if is_raining and command.water_litres == 0.0:
            reward += self.rain_bonus

        # Pre-waterlogging penalty: penalise irrigating when moisture is already
        # above optimal_max during rain. Scales 0→1 between optimal_max and
        # field_capacity. Teaches the agent to stop before reaching the death zone.
        if is_raining and command.water_litres > 0.0 and soil_moisture_pct > t.optimal_max:
            danger_range = max(1.0, t.field_capacity - t.optimal_max)
            danger_ratio = min(1.0, (soil_moisture_pct - t.optimal_max) / danger_range)
            reward -= self.overwater_penalty * danger_ratio * water_fraction

        # Emergency penalty: soil critically dry but agent applied too little.
        if (
            soil_moisture_pct <= t.wilting_point
            and command.water_litres < self.zone.emergency_min_litres
        ):
            reward -= 5.0

        return round(reward, 4)
