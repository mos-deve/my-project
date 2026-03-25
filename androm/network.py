"""
Network - Connects thousands of units together.
Manages signal propagation through the unit graph.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional
from androm.unit import Unit, UnitType


@dataclass
class Network:
    """
    A network of interconnected computational units.
    
    Signals flow from input units through hidden units to output units.
    The network processes data by propagating signals through the graph.
    """
    
    units: dict[int, Unit] = field(default_factory=dict)
    input_ids: list[int] = field(default_factory=list)
    output_ids: list[int] = field(default_factory=list)
    _next_id: int = 0
    
    def add_unit(self, unit_type: UnitType, inputs: list[int] | None = None,
                 weights: list[float] | None = None, bias: float = 0.0,
                 noise: float = 0.0) -> int:
        """Add a unit to the network. Returns the unit's ID."""
        unit_id = self._next_id
        self._next_id += 1
        
        unit = Unit(
            id=unit_id,
            unit_type=unit_type,
            inputs=inputs or [],
            weights=weights or ([1.0] * len(inputs) if inputs else []),
            bias=bias,
            noise_factor=noise,
        )
        self.units[unit_id] = unit
        return unit_id
    
    def add_input_unit(self) -> int:
        """Add an input unit (passes value through)."""
        uid = self.add_unit(UnitType.MATH, inputs=[], weights=[])
        self.input_ids.append(uid)
        return uid
    
    def add_output_unit(self, inputs: list[int], weights: list[float] | None = None) -> int:
        """Add an output unit."""
        uid = self.add_unit(UnitType.MATH, inputs=inputs, weights=weights)
        self.output_ids.append(uid)
        return uid
    
    def propagate(self, inputs: list[float]) -> list[float]:
        """
        Forward pass: propagate inputs through the network.
        
        Args:
            inputs: Values for input units
            
        Returns:
            Values from output units
        """
        if len(inputs) != len(self.input_ids):
            raise ValueError(f"Expected {len(self.input_ids)} inputs, got {len(inputs)}")
        
        # Set input values
        values: dict[int, float] = {}
        for uid, val in zip(self.input_ids, inputs):
            values[uid] = val
            self.units[uid]._output = val
        
        # Topological propagation (simple: iterate until stable)
        # For DAGs, this converges in one pass
        max_iterations = len(self.units) + 1
        for _ in range(max_iterations):
            changed = False
            for uid, unit in self.units.items():
                if uid in self.input_ids:
                    continue
                
                # Gather input values
                input_vals = []
                all_ready = True
                for inp_id in unit.inputs:
                    if inp_id in values:
                        input_vals.append(values[inp_id])
                    else:
                        all_ready = False
                        break
                
                if all_ready and input_vals:
                    new_val = unit.compute(input_vals)
                    if uid not in values or abs(values[uid] - new_val) > 1e-10:
                        values[uid] = new_val
                        changed = True
            
            if not changed:
                break
        
        # Collect output values
        return [values.get(uid, 0.0) for uid in self.output_ids]
    
    def reset(self):
        """Reset all units."""
        for unit in self.units.values():
            unit.reset()
    
    def size(self) -> int:
        """Number of units in the network."""
        return len(self.units)
    
    def random_connect(self, num_units: int, connectivity: float = 0.1):
        """
        Add random units with random connections.
        
        Args:
            num_units: Number of units to add
            connectivity: Probability of connecting to each existing unit
        """
        existing = list(self.units.keys())
        
        for _ in range(num_units):
            # Pick random type
            unit_type = random.choice(list(UnitType))
            
            # Random connections
            inputs = [uid for uid in existing if random.random() < connectivity]
            if not inputs and existing:
                inputs = [random.choice(existing)]
            
            weights = [random.uniform(-1, 1) for _ in inputs]
            bias = random.uniform(-1, 1)
            noise = random.uniform(0, 0.1) if unit_type == UnitType.PROBABILISTIC else 0.0
            
            self.add_unit(unit_type, inputs, weights, bias, noise)
            existing.append(self._next_id - 1)
    
    def to_dict(self) -> dict:
        """Serialize network."""
        return {
            "units": [u.to_dict() for u in self.units.values()],
            "input_ids": self.input_ids,
            "output_ids": self.output_ids,
            "next_id": self._next_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Network":
        """Deserialize network."""
        net = cls()
        net._next_id = data.get("next_id", 0)
        net.input_ids = data.get("input_ids", [])
        net.output_ids = data.get("output_ids", [])
        for udata in data.get("units", []):
            unit = Unit.from_dict(udata)
            net.units[unit.id] = unit
        return net
