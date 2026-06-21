"""Plant vitality tracker for the irrigation RL agent.

Tracks plant survival each step using stage-specific death thresholds.
If the plant stays below its critical moisture threshold for too many
consecutive hours, vitality drops to zero and the plant dies.

Key agronomic facts modelled:
  - Seedlings (Stage 0) die within 2 dry hours — no root system yet
  - Stage 2 (Flowering) death threshold is STRESS THRESHOLD (55%), not
    wilting point — flower drop at that level = total crop failure
  - Waterlogging above field capacity causes root suffocation (slower death)
  - Earlier death = larger penalty (wasted entire season investment)
"""

from __future__ import annotations

from dataclasses import dataclass

from irrigation.config_loader import load_config
from irrigation.crops.base import CropProfile

# Load all vitality parameters from config.yaml → stages section
_cfg = load_config()
_s   = _cfg["stages"]     # stage-level config block
_dp  = _cfg["reward"]["death_penalty"]  # death penalties block

# Map stage index to YAML key name
_STAGE_KEYS = {
    0: "germination",
    1: "vegetative",
    2: "flowering",
    3: "fruit_development",
    4: "maturity",
}

# ---------------------------------------------------------------------------
# Stage-specific vitality configuration — loaded from config.yaml
# Edit values in config.yaml, they reflect here automatically.
# ---------------------------------------------------------------------------

@dataclass
class VitalityConfig:
    dry_threshold_type: str   # "wilting_point" or "stress_threshold"
    dry_hours_limit: int      # consecutive hours below dry threshold → dead
    wet_hours_limit: int      # consecutive hours above field capacity → dead
    vitality_drain: float     # vitality lost per dry hour (0.0–1.0)


# Built from config.yaml stages section — change values there, not here
VITALITY_CONFIGS: dict[int, VitalityConfig] = {
    idx: VitalityConfig(
        dry_threshold_type=_s[key]["dry_threshold"],
        dry_hours_limit=_s[key]["dry_hours_limit"],
        wet_hours_limit=_s[key]["wet_hours_limit"],
        vitality_drain=_s[key]["vitality_drain"],
    )
    for idx, key in _STAGE_KEYS.items()
}

# Terminal death penalty per stage — loaded from config.yaml reward section
DEATH_PENALTY: dict[int, float] = {
    0: float(_dp["stage_0"]),
    1: float(_dp["stage_1"]),
    2: float(_dp["stage_2"]),
    3: float(_dp["stage_3"]),
    4: float(_dp["stage_4"]),
}


class PlantVitalityTracker:
    """Tracks plant survival using stage-specific critical moisture thresholds.

    Vitality (0.0 → 1.0):
        1.0 = perfectly healthy
        0.5 = stressed but alive
        0.0 = dead → episode terminates

    Args:
        crop: Crop profile providing stage-specific moisture thresholds.
    """

    def __init__(self, crop: CropProfile) -> None:
        self.crop = crop
        self.vitality: float = 1.0
        self.is_dead: bool = False
        self._consecutive_dry: int = 0
        self._consecutive_wet: int = 0
        self._death_cause: str = ""
        self._death_stage: int = -1

    def reset(self) -> None:
        """Reset for a new episode."""
        self.vitality = 1.0
        self.is_dead = False
        self._consecutive_dry = 0
        self._consecutive_wet = 0
        self._death_cause = ""
        self._death_stage = -1

    def update(self, moisture_pct: float, stage: int) -> None:
        """Update vitality for one hourly step.

        Args:
            moisture_pct: Current soil moisture percentage.
            stage: Current dynamic growth stage (0-4).
        """
        if self.is_dead:
            return

        t   = self.crop.moisture_thresholds_for_stage(stage)
        cfg = VITALITY_CONFIGS[stage]

        # Determine the dry threshold for this stage
        dry_threshold = (
            t.stress_threshold
            if cfg.dry_threshold_type == "stress_threshold"
            else t.wilting_point
        )

        if moisture_pct < dry_threshold:
            # Drought stress — drain vitality
            self._consecutive_dry += 1
            self._consecutive_wet = 0
            self.vitality = max(0.0, self.vitality - cfg.vitality_drain)

            if self._consecutive_dry >= cfg.dry_hours_limit or self.vitality <= 0.0:
                self.vitality = 0.0
                self.is_dead = True
                self._death_cause = "drought"
                self._death_stage = stage

        elif moisture_pct > t.field_capacity:
            # Waterlogging — drain vitality at half rate (slower death)
            self._consecutive_wet += 1
            self._consecutive_dry = 0
            self.vitality = max(0.0, self.vitality - cfg.vitality_drain * 0.5)

            if self._consecutive_wet >= cfg.wet_hours_limit or self.vitality <= 0.0:
                self.vitality = 0.0
                self.is_dead = True
                self._death_cause = "waterlogging"
                self._death_stage = stage

        else:
            # Healthy range — reset counters and slowly recover
            self._consecutive_dry = 0
            self._consecutive_wet = 0
            self.vitality = min(1.0, self.vitality + 0.01)

    @property
    def death_penalty(self) -> float:
        """Return the terminal death penalty for the stage the plant died in."""
        if not self.is_dead:
            return 0.0
        return DEATH_PENALTY.get(self._death_stage, -50.0)

    @property
    def consecutive_wet_hours(self) -> int:
        """Current count of consecutive hours above field capacity."""
        return self._consecutive_wet

    def death_info(self) -> dict:
        """Return death event details for the info dict."""
        return {
            "plant_dead":   self.is_dead,
            "death_cause":  self._death_cause,
            "death_stage":  self._death_stage,
            "vitality":     round(self.vitality, 4),
        }
