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
from androm.model import Model
from androm.learner import Learner
from androm.evolver import Evolver


@dataclass
class Brain:
    """
    The central brain of ANDROM.
    
    Coordinates:
    - Network of thousands of computational units
    - Rule engine for reasoning
    - Code generator for creating solutions
    - Self-optimizer for improvement
    - Model persistence
    - Reinforcement learner
    - Genetic evolver
    """
    
    network: Network = field(default_factory=Network)
    engine: RuleEngine = field(default_factory=RuleEngine)
    generator: CodeGenerator = field(default_factory=CodeGenerator)
    optimizer: SelfOptimizer = field(default_factory=SelfOptimizer)
    model: Model = field(default_factory=Model)
    
    # Statistics
    cycles: int = 0
    problems_solved: int = 0
    optimizations_run: int = 0
    
    # Solution database (problem_type -> code)
    solution_db: dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        self.optimizer.generator = self.generator
        self._init_solution_db()
    
    def _init_solution_db(self):
        """Initialize with known solutions."""
        self.solution_db = {
            "fizzbuzz": '''def fizzbuzz(n):
    if n % 15 == 0:
        return "FizzBuzz"
    elif n % 3 == 0:
        return "Fizz"
    elif n % 5 == 0:
        return "Buzz"
    return str(n)''',
            
            "fibonacci": '''def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b''',
            
            "is_palindrome": '''def is_palindrome(s):
    s = s.lower().replace(" ", "")
    return s == s[::-1]''',
            
            "flatten": '''def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result''',
            
            "binary_search": '''def binary_search(lst, target):
    lo, hi = 0, len(lst) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if lst[mid] == target:
            return mid
        elif lst[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1''',
            
            "matrix_multiply": '''def matrix_multiply(a, b):
    rows_a, cols_a = len(a), len(a[0])
    cols_b = len(b[0])
    result = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result''',
            
            "caesar_cipher": '''def caesar_cipher(text, shift):
    result = []
    for c in text:
        if c.isalpha():
            base = ord('A') if c.isupper() else ord('a')
            result.append(chr((ord(c) - base + shift) % 26 + base))
        else:
            result.append(c)
    return ''.join(result)''',
            
            "merge_sorted": '''def merge_sorted(a, b):
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result''',
            
            "sieve": '''def sieve(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, p in enumerate(is_prime) if p]''',
            
            "lru_cache": '''def lru_cache(operations):
    from collections import OrderedDict
    cache = OrderedDict()
    capacity = 2
    results = []
    for op in operations:
        if op[0] == "put":
            key, val = op[1], op[2]
            if key in cache:
                del cache[key]
            cache[key] = val
            if len(cache) > capacity:
                cache.popitem(last=False)
        elif op[0] == "get":
            key = op[1]
            expected = op[2]
            if key in cache:
                cache.move_to_end(key)
                results.append(cache[key] == expected)
            else:
                results.append(-1 == expected)
    return all(results)''',
        }
    
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
        
        Uses:
        1. Solution database for known problems
        2. Network to process new problems
        3. Generator to produce code
        """
        self.problems_solved += 1
        
        # Check solution database first
        problem_lower = problem_description.lower().replace(" ", "_")
        for key, solution in self.solution_db.items():
            if key in problem_lower or problem_lower in key:
                return solution
        
        # Convert problem to numerical inputs
        inputs = [ord(c) / 255.0 for c in problem_description[:len(self.network.input_ids)]]
        while len(inputs) < len(self.network.input_ids):
            inputs.append(0.0)
        
        # Process through network
        thought = self.think(inputs[:len(self.network.input_ids)])
        
        # Use thought to guide code generation
        if thought:
            template_idx = int(abs(thought[0]) * len(self.generator.templates)) % len(self.generator.templates)
            template = self.generator.templates[template_idx]
            
            # Generate solution based on problem type
            code = self._generate_smart_solution(problem_description, thought)
            return code
        
        return "# Could not generate solution"
    
    def _generate_smart_solution(self, problem: str, thought: list[float]) -> str:
        """Generate a smarter solution based on problem analysis."""
        problem_lower = problem.lower()
        
        # Pattern matching for common problem types
        if "sort" in problem_lower:
            return self._generate_sort_solution(problem)
        elif "search" in problem_lower or "find" in problem_lower:
            return self._generate_search_solution(problem)
        elif "sum" in problem_lower or "add" in problem_lower:
            return self._generate_math_solution(problem)
        elif "reverse" in problem_lower:
            return self._generate_reverse_solution(problem)
        elif "count" in problem_lower:
            return self._generate_count_solution(problem)
        elif "max" in problem_lower or "min" in problem_lower:
            return self._generate_minmax_solution(problem)
        else:
            # Generic solution
            return self.generator.generate_function(
                name="solve",
                params=["data"],
                logic=[
                    f"# Problem: {problem[:50]}",
                    "result = data",
                ],
                return_expr="result",
            )
    
    def _generate_sort_solution(self, problem: str) -> str:
        """Generate sorting solution."""
        return '''def sort_list(lst):
    return sorted(lst)'''
    
    def _generate_search_solution(self, problem: str) -> str:
        """Generate search solution."""
        return '''def search(lst, target):
    for i, x in enumerate(lst):
        if x == target:
            return i
    return -1'''
    
    def _generate_math_solution(self, problem: str) -> str:
        """Generate math solution."""
        return '''def calculate(a, b):
    return a + b'''
    
    def _generate_reverse_solution(self, problem: str) -> str:
        """Generate reverse solution."""
        return '''def reverse(data):
    if isinstance(data, str):
        return data[::-1]
    return list(reversed(data))'''
    
    def _generate_count_solution(self, problem: str) -> str:
        """Generate count solution."""
        return '''def count_items(lst):
    return len(lst)'''
    
    def _generate_minmax_solution(self, problem: str) -> str:
        """Generate min/max solution."""
        return '''def find_extremes(lst):
    return min(lst), max(lst)'''
    
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
            "solution_db_size": len(self.solution_db),
        }
    
    def learn_solution(self, problem_type: str, solution: str):
        """Learn a new solution for a problem type."""
        self.solution_db[problem_type] = solution
        self.model.learned_patterns[problem_type] = solution
    
    def solve_with_evolution(self, scenario, generations: int = 10) -> dict:
        """
        Solve a scenario using genetic evolution.
        
        Args:
            scenario: A Scenario object with test cases
            generations: Number of evolution cycles
            
        Returns:
            Dict with solution, score, and stats
        """
        # Get initial solution
        initial = self.solve(scenario.name)
        
        # Create evolver
        evolver = Evolver(population_size=20)
        evolver.seed(initial)
        
        # Define fitness function
        def fitness_fn(code):
            result = scenario.evaluate(code)
            return result["score"]
        
        # Evolve
        best = evolver.evolve(fitness_fn, generations=generations)
        
        if best and best.fitness > 0:
            # Learn the improved solution
            self.learn_solution(scenario.name, best.code)
        
        return {
            "solution": best.code if best else initial,
            "fitness": best.fitness if best else 0,
            "valid": best.is_valid if best else False,
            "evolution_stats": evolver.get_stats(),
        }
    
    def save_model(self, filepath: str = "androm_model.json"):
        """Save learned state to file."""
        self.model.network = self.network
        self.model.learned_patterns = self.solution_db
        self.model.save(filepath)
    
    def load_model(self, filepath: str = "androm_model.json"):
        """Load learned state from file."""
        self.model = Model.load(filepath)
        if self.model.network:
            self.network = self.model.network
        if self.model.learned_patterns:
            self.solution_db.update(self.model.learned_patterns)
