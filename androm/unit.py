"""
Unit - The fundamental computational element in ANDROM.
Each unit is a small rule-based processor that does math operations.
"""

from __future__ import annotations
import math
import random
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Callable


class UnitType(Enum):
    """Types of computational units."""
    MATH = auto()       # Basic arithmetic
    LOGIC = auto()      # Boolean logic
    COMPARE = auto()    # Comparison operations
    AGGREGATE = auto()  # Aggregation (sum, mean, etc.)
    TRANSFORM = auto()  # Data transformation
    GATE = auto()       # Conditional gating
    MEMORY = auto()     # Stateful memory unit
    PROBABILISTIC = auto()  # Probabilistic decisions


@dataclass
class Unit:
    """
    A single computational unit.
    
    Each unit:
    - Receives inputs from other units or external sources
    - Applies a deterministic rule (math operation)
    - Optionally adds probabilistic noise
    - Produces an output signal
    """
    
    id: int
    unit_type: UnitType
    inputs: list[int] = field(default_factory=list)  # IDs of input units
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    noise_factor: float = 0.0  # 0.0 = fully deterministic
    _output: float = 0.0
    _memory: float = 0.0  # For MEMORY type units
    
    def __post_init__(self):
        if not self.weights and self.inputs:
            self.weights = [1.0] * len(self.inputs)
    
    def compute(self, input_values: list[float]) -> float:
        """Compute output given input values."""
        if len(input_values) != len(self.weights):
            raise ValueError(f"Expected {len(self.weights)} inputs, got {len(input_values)}")
        
        # Weighted sum
        weighted_sum = sum(v * w for v, w in zip(input_values, self.weights)) + self.bias
        
        # Apply unit-type specific operation
        result = self._apply_operation(weighted_sum, input_values)
        
        # Add probabilistic noise if enabled
        if self.noise_factor > 0:
            result += random.gauss(0, self.noise_factor)
        
        self._output = result
        return result
    
    def _apply_operation(self, weighted_sum: float, inputs: list[float]) -> float:
        """Apply the unit's specific operation based on its type."""
        match self.unit_type:
            case UnitType.MATH:
                # Clamp to prevent overflow, apply tanh activation
                return math.tanh(weighted_sum)
            
            case UnitType.LOGIC:
                # Binary step: 1 if positive, 0 otherwise
                return 1.0 if weighted_sum > 0 else 0.0
            
            case UnitType.COMPARE:
                # Compare first two inputs
                if len(inputs) >= 2:
                    return 1.0 if inputs[0] > inputs[1] else 0.0
                return 1.0 if weighted_sum > 0 else 0.0
            
            case UnitType.AGGREGATE:
                # Mean of inputs
                return sum(inputs) / len(inputs) if inputs else 0.0
            
            case UnitType.TRANSFORM:
                # Sigmoid transformation
                try:
                    return 1.0 / (1.0 + math.exp(-weighted_sum))
                except OverflowError:
                    return 0.0 if weighted_sum < 0 else 1.0
            
            case UnitType.GATE:
                # First input controls whether second input passes
                if len(inputs) >= 2:
                    return inputs[1] if inputs[0] > 0.5 else 0.0
                return weighted_sum
            
            case UnitType.MEMORY:
                # Remembers previous value, blends with new input
                alpha = 0.5  # Memory decay
                self._memory = alpha * self._memory + (1 - alpha) * weighted_sum
                return self._memory
            
            case UnitType.PROBABILISTIC:
                # Probabilistic output based on weighted_sum
                prob = 1.0 / (1.0 + math.exp(-weighted_sum))
                return 1.0 if random.random() < prob else 0.0
            
            case _:
                return weighted_sum
    
    @property
    def output(self) -> float:
        return self._output
    
    def reset(self):
        """Reset unit state."""
        self._output = 0.0
        self._memory = 0.0
    
    def to_dict(self) -> dict:
        """Serialize unit to dictionary."""
        return {
            "id": self.id,
            "type": self.unit_type.name,
            "inputs": self.inputs,
            "weights": self.weights,
            "bias": self.bias,
            "noise": self.noise_factor,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Unit":
        """Deserialize unit from dictionary."""
        return cls(
            id=data["id"],
            unit_type=UnitType[data["type"]],
            inputs=data.get("inputs", []),
            weights=data.get("weights", []),
            bias=data.get("bias", 0.0),
            noise_factor=data.get("noise", 0.0),
        )
