"""Tests for actuator implementations."""

from __future__ import annotations

import pytest

from irrigation.actuators.base import IrrigationCommand
from irrigation.actuators.simulation import SimulatedActuator
from irrigation.sensors.simulation import SimulatedSoilMoistureSensor


class TestIrrigationCommand:
    def test_zero_water_is_no_irrigation(self):
        cmd = IrrigationCommand(water_litres=0.0)
        assert cmd.water_litres == 0.0

    def test_positive_water(self):
        cmd = IrrigationCommand(water_litres=5.0)
        assert cmd.water_litres == 5.0

    def test_large_irrigation(self):
        cmd = IrrigationCommand(water_litres=15.0)
        assert cmd.water_litres == 15.0


class TestSimulatedActuator:
    def test_no_op_on_zero_litres(self):
        actuator = SimulatedActuator()
        actuator.execute(IrrigationCommand(water_litres=0.0))
        assert actuator.irrigation_count() == 0

    def test_irrigation_recorded(self):
        actuator = SimulatedActuator()
        actuator.execute(IrrigationCommand(water_litres=5.0))
        assert actuator.irrigation_count() == 1

    def test_water_tracking(self):
        actuator = SimulatedActuator()
        actuator.execute(IrrigationCommand(water_litres=8.0))
        assert actuator.total_water_used_litres == pytest.approx(8.0)

    def test_multiple_irrigations_accumulate(self):
        actuator = SimulatedActuator()
        actuator.execute(IrrigationCommand(water_litres=5.0))
        actuator.execute(IrrigationCommand(water_litres=5.0))
        assert actuator.irrigation_count() == 2
        assert actuator.total_water_used_litres == pytest.approx(10.0)

    def test_reset_clears_history(self):
        actuator = SimulatedActuator()
        actuator.execute(IrrigationCommand(water_litres=5.0))
        actuator.reset()
        assert actuator.irrigation_count() == 0
        assert actuator.total_water_used_litres == 0.0

    def test_updates_soil_sensor(self):
        soil = SimulatedSoilMoistureSensor(initial_moisture_pct=30.0, seed=42)
        actuator = SimulatedActuator(soil_sensor=soil)
        before = soil.moisture_pct
        actuator.execute(IrrigationCommand(water_litres=10.0))
        assert soil.moisture_pct > before

    def test_not_active_before_execute(self):
        actuator = SimulatedActuator()
        assert not actuator.is_active

    def test_stop_does_not_raise(self):
        actuator = SimulatedActuator()
        actuator.stop()
