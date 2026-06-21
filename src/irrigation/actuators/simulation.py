"""Simulated actuator for testing and development."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from irrigation.actuators.base import ActuatorInterface, IrrigationCommand
from irrigation.zone_config import ZoneConfig


@dataclass
class IrrigationEvent:
    """Records a single irrigation event."""

    timestamp: datetime
    water_litres: float


class SimulatedActuator(ActuatorInterface):
    """An actuator that simulates irrigation without hardware.

    Notifies an optional soil sensor so the RL environment sees a realistic
    moisture response to irrigation commands.

    Args:
        soil_sensor: Optional simulated soil sensor to update on irrigation.
        zone: Zone configuration used to derive moisture_per_litre. Defaults
            to ZoneConfig() so the value always reflects the current config.
    """

    def __init__(
        self,
        soil_sensor: object | None = None,
        zone: ZoneConfig | None = None,
    ) -> None:
        self.soil_sensor = soil_sensor
        self.zone = zone or ZoneConfig()
        self._active: bool = False
        self.history: list[IrrigationEvent] = []
        self.total_water_used_litres: float = 0.0

    def execute(self, command: IrrigationCommand) -> None:
        if command.water_litres <= 0.0:
            return

        self._active = True
        self.history.append(
            IrrigationEvent(
                timestamp=datetime.now(),
                water_litres=command.water_litres,
            )
        )
        self.total_water_used_litres += command.water_litres

        if self.soil_sensor is not None and hasattr(self.soil_sensor, "irrigate"):
            self.soil_sensor.irrigate(command.water_litres * self.zone.moisture_per_litre)

        self._active = False

    def stop(self) -> None:
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def irrigation_count(self) -> int:
        """Total number of irrigation events recorded."""
        return len(self.history)

    def reset(self) -> None:
        """Reset recorded history and water usage counter."""
        self.history.clear()
        self.total_water_used_litres = 0.0
