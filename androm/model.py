"""
Model - Persistence layer for ANDROM.
Saves and loads learned state (network, rules, patterns, history).
"""

from __future__ import annotations
import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Any
from androm.network import Network
from androm.engine import RuleEngine
from androm.generator import CodeGenerator


@dataclass
class Model:
    """
    Persistent model that stores learned state.
    
    Saves:
    - Network weights and connections
    - Learned code patterns
    - Solution history
    - Performance metrics
    """
    
    version: str = "1.0"
    network: Network | None = None
    learned_patterns: dict[str, str] = field(default_factory=dict)
    solution_history: list[dict] = field(default_factory=list)
    performance: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def save(self, filepath: str):
        """Save model to file."""
        data = {
            "version": self.version,
            "network": self.network.to_dict() if self.network else None,
            "learned_patterns": self.learned_patterns,
            "solution_history": self.solution_history,
            "performance": self.performance,
            "metadata": self.metadata,
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "Model":
        """Load model from file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        model = cls(
            version=data.get("version", "1.0"),
            learned_patterns=data.get("learned_patterns", {}),
            solution_history=data.get("solution_history", []),
            performance=data.get("performance", {}),
            metadata=data.get("metadata", {}),
        )
        
        if data.get("network"):
            model.network = Network.from_dict(data["network"])
        
        return model
    
    def save_binary(self, filepath: str):
        """Save model in binary format (faster, smaller)."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_binary(cls, filepath: str) -> "Model":
        """Load model from binary file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
    
    def record_solution(self, problem: str, solution: str, 
                        score: float, execution_time: float):
        """Record a solution attempt."""
        self.solution_history.append({
            "problem": problem,
            "solution_hash": hash(solution),
            "score": score,
            "execution_time": execution_time,
            "solution_length": len(solution.splitlines()),
        })
    
    def get_best_solutions(self, n: int = 10) -> list[dict]:
        """Get top N solutions by score."""
        sorted_history = sorted(
            self.solution_history, 
            key=lambda x: x.get("score", 0), 
            reverse=True
        )
        return sorted_history[:n]
    
    def update_performance(self, metric: str, value: float):
        """Update a performance metric."""
        if metric in self.performance:
            # Exponential moving average
            self.performance[metric] = 0.9 * self.performance[metric] + 0.1 * value
        else:
            self.performance[metric] = value
