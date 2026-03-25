"""
Brain - The main orchestrator for ANDROM.
Ties together network, rules, code generation, and self-optimization.
"""

from __future__ import annotations
import random
import time
from dataclasses import dataclass, field
from typing import Any
from androm.unit import Unit, UnitType
from androm.network import Network
from androm.engine import RuleEngine
from androm.generator import CodeGenerator
from androm.optimizer import SelfOptimizer


@dataclass
class Brain:
    """
    The central brain of ANDROM.
    
    Coordinates:
    - Network of thousands of computational units
    - Rule engine for reasoning
    - Code generator for creating solutions
    - Self-optimizer for improvement
    """
    
    network: Network = field(default_factory=Network)
    engine: RuleEngine = field(default_factory=RuleEngine)
    generator: CodeGenerator = field(default_factory=CodeGenerator)
    optimizer: SelfOptimizer = field(default_factory=SelfOptimizer)
    
    # Statistics
    cycles: int = 0
    problems_solved: int = 0
    optimizations_run: int = 0
    
    def __post_init__(self):
        self.optimizer.generator = self.generator
    
    def build_network(self, num_units: int = 1000, connectivity: float = 0.05):
        """
        Build a network with specified number of units.
        
        Args:
            num_units: Total units to create
            connectivity: How connected each unit is (0-1)
        """
        # Create input units (10% of total)
        num_inputs = max(1, num_units // 10)
        for _ in range(num_inputs):
            self.network.add_input_unit()
        
        # Create hidden units with random connections
        self.network.random_connect(num_units - num_inputs - num_inputs, connectivity)
        
        # Create output units (10% of total)
        num_outputs = max(1, num_units // 10)
        existing = list(self.network.units.keys())
        for _ in range(num_outputs):
            inputs = random.sample(existing, min(5, len(existing)))
            weights = [random.uniform(-1, 1) for _ in inputs]
            self.network.add_output_unit(inputs, weights)
        
        return self
    
    def think(self, inputs: list[float]) -> list[float]:
        """Process inputs through the network."""
        self.cycles += 1
        return self.network.propagate(inputs)
    
    def reason(self, facts: dict[str, Any]) -> list[str]:
        """Apply rules to facts."""
        for key, value in facts.items():
            self.engine.set_fact(key, value)
        return self.engine.run()
    
    def solve(self, problem_description: str) -> str:
        """
        Attempt to solve a general problem.
        
        Uses network to process the problem, rules to reason about it,
        and generator to produce code.
        """
        self.problems_solved += 1
        
        # Convert problem to numerical inputs (simple hash-based)
        inputs = [ord(c) / 255.0 for c in problem_description[:len(self.network.input_ids)]]
        # Pad if needed
        while len(inputs) < len(self.network.input_ids):
            inputs.append(0.0)
        
        # Process through network
        thought = self.think(inputs[:len(self.network.input_ids)])
        
        # Use thought to guide code generation
        # Map thought values to template selection
        if thought:
            template_idx = int(abs(thought[0]) * len(self.generator.templates)) % len(self.generator.templates)
            template = self.generator.templates[template_idx]
            
            # Generate basic solution structure
            code = self.generator.generate_function(
                name="solve",
                params=["input_data"],
                logic=[
                    f"# Solution for: {problem_description[:50]}...",
                    "result = input_data",
                ],
                return_expr="result",
            )
            return code
        
        return "# Could not generate solution"
    
    def optimize_self(self) -> dict:
        """
        Run self-optimization on ANDROM's own code.
        The key self-improvement function.
        """
        self.optimizations_run += 1
        
        results = {}
        modules = [
            ("unit", "androm.unit"),
            ("network", "androm.network"),
            ("engine", "androm.engine"),
            ("generator", "androm.generator"),
            ("optimizer", "androm.optimizer"),
        ]
        
        for name, module_path in modules:
            try:
                import importlib
                mod = importlib.import_module(module_path)
                result = self.optimizer.optimize_module(mod)
                results[name] = {
                    "original_lines": result.original_lines,
                    "optimized_lines": result.optimized_lines,
                    "reduction": f"{result.reduction_percent:.1f}%",
                    "valid": result.is_valid,
                    "improved": result.improved,
                }
            except Exception as e:
                results[name] = {"error": str(e)}
        
        return results
    
    def run_cycle(self) -> dict:
        """
        Run one complete ANDROM cycle:
        1. Generate random inputs
        2. Process through network
        3. Apply rules
        4. Generate code
        5. Optionally optimize
        """
        # Random inputs
        inputs = [random.uniform(-1, 1) for _ in self.network.input_ids]
        
        # Process
        outputs = self.think(inputs)
        
        # Reason about outputs
        facts = {f"output_{i}": v for i, v in enumerate(outputs)}
        facts["cycle"] = self.cycles
        fired_rules = self.reason(facts)
        
        # Generate code based on patterns
        code = self.solve(f"cycle_{self.cycles}")
        
        return {
            "cycle": self.cycles,
            "inputs": inputs[:5],  # Sample
            "outputs": outputs[:5],  # Sample
            "rules_fired": fired_rules,
            "code_lines": len(code.splitlines()),
        }
    
    def status(self) -> dict:
        """Get brain status."""
        return {
            "units": self.network.size(),
            "cycles": self.cycles,
            "problems_solved": self.problems_solved,
            "optimizations": self.optimizations_run,
            "optimizer_stats": self.optimizer.get_stats(),
            "rules": len(self.engine.rules),
        }
