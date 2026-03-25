"""
Evolver - Genetic algorithm for code evolution.
Evolves code solutions through mutation, crossover, and selection.
"""

from __future__ import annotations
import ast
import random
import copy
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class CodeOrganism:
    """A single code solution that can evolve."""
    code: str
    fitness: float = 0.0
    generation: int = 0
    parent_ids: list[int] = field(default_factory=list)
    mutations: list[str] = field(default_factory=list)
    _id: int = field(default_factory=lambda: random.randint(0, 2**31))
    
    @property
    def id(self) -> int:
        return self._id
    
    @property
    def lines(self) -> int:
        return len(self.code.strip().splitlines())
    
    @property
    def is_valid(self) -> bool:
        try:
            ast.parse(self.code)
            return True
        except SyntaxError:
            return False


class Evolver:
    """
    Genetic algorithm for evolving code solutions.
    
    Operations:
    - Mutation: Randomly modify code
    - Crossover: Combine two solutions
    - Selection: Keep best solutions
    """
    
    def __init__(self, population_size: int = 20):
        self.population_size = population_size
        self.population: list[CodeOrganism] = []
        self.best_ever: CodeOrganism | None = None
        self.generation: int = 0
        self.history: list[dict] = []
    
    def seed(self, code: str):
        """Seed initial population from a code template."""
        self.population = []
        for i in range(self.population_size):
            org = CodeOrganism(
                code=self._mutate_code(code),
                generation=0,
            )
            self.population.append(org)
    
    def evolve(self, fitness_fn: Callable[[str], float], 
               generations: int = 10) -> CodeOrganism | None:
        """
        Evolve population for N generations.
        
        Args:
            fitness_fn: Function that scores code (higher is better)
            generations: Number of generations to evolve
            
        Returns:
            Best organism found
        """
        for gen in range(generations):
            self.generation = gen
            
            # Evaluate fitness
            for org in self.population:
                if org.is_valid:
                    org.fitness = fitness_fn(org.code)
                else:
                    org.fitness = -1000.0  # Heavy penalty for invalid code
            
            # Sort by fitness
            self.population.sort(key=lambda o: o.fitness, reverse=True)
            
            # Track best
            if self.best_ever is None or self.population[0].fitness > self.best_ever.fitness:
                self.best_ever = copy.deepcopy(self.population[0])
            
            # Record history
            self.history.append({
                "generation": gen,
                "best_fitness": self.population[0].fitness,
                "avg_fitness": sum(o.fitness for o in self.population) / len(self.population),
                "valid_count": sum(1 for o in self.population if o.is_valid),
            })
            
            # Selection: keep top 50%
            survivors = self.population[:self.population_size // 2]
            
            # Create next generation
            next_gen = []
            
            # Elitism: keep best
            next_gen.append(copy.deepcopy(survivors[0]))
            
            # Fill rest with mutations and crossovers
            while len(next_gen) < self.population_size:
                if random.random() < 0.3 and len(survivors) >= 2:
                    # Crossover
                    p1, p2 = random.sample(survivors, 2)
                    child_code = self._crossover(p1.code, p2.code)
                    mutations = ["crossover"]
                else:
                    # Mutation
                    parent = random.choice(survivors)
                    child_code = self._mutate_code(parent.code)
                    mutations = ["mutation"]
                
                child = CodeOrganism(
                    code=child_code,
                    generation=gen + 1,
                    parent_ids=[s.id for s in survivors[:2]],
                    mutations=mutations,
                )
                next_gen.append(child)
            
            self.population = next_gen
        
        return self.best_ever
    
    def _mutate_code(self, code: str) -> str:
        """Apply random mutations to code."""
        lines = code.splitlines()
        if not lines:
            return code
        
        mutation_type = random.choice([
            "add_line", "remove_line", "modify_line",
            "swap_lines", "duplicate_line", "simplify",
        ])
        
        if mutation_type == "add_line" and len(lines) > 0:
            # Add a new line at random position
            pos = random.randint(0, len(lines))
            new_lines = self._generate_lines(1)
            lines.insert(pos, new_lines[0])
        
        elif mutation_type == "remove_line" and len(lines) > 2:
            # Remove a non-def/class line
            removable = [i for i, l in enumerate(lines) 
                        if not l.strip().startswith(("def ", "class ", "return"))]
            if removable:
                lines.pop(random.choice(removable))
        
        elif mutation_type == "modify_line":
            # Modify a line
            pos = random.randint(0, len(lines) - 1)
            lines[pos] = self._mutate_line(lines[pos])
        
        elif mutation_type == "swap_lines" and len(lines) > 2:
            # Swap two lines
            i, j = random.sample(range(len(lines)), 2)
            lines[i], lines[j] = lines[j], lines[i]
        
        elif mutation_type == "duplicate_line":
            # Duplicate a line
            pos = random.randint(0, len(lines) - 1)
            lines.insert(pos, lines[pos])
        
        elif mutation_type == "simplify":
            # Simplify a line
            pos = random.randint(0, len(lines) - 1)
            lines[pos] = self._simplify_line(lines[pos])
        
        return "\n".join(lines)
    
    def _mutate_line(self, line: str) -> str:
        """Mutate a single line of code."""
        stripped = line.strip()
        indent = line[:len(line) - len(line.lstrip())]
        
        # Simple mutations
        mutations = [
            lambda l: l.replace("True", "False") if "True" in l else l,
            lambda l: l.replace("False", "True") if "False" in l else l,
            lambda l: l.replace("+", "-") if "+" in l else l,
            lambda l: l.replace("-", "+") if "-" in l else l,
            lambda l: l.replace("*", "//") if "*" in l else l,
            lambda l: l.replace("==", "!=") if "==" in l else l,
            lambda l: l.replace("!=", "==") if "!=" in l else l,
            lambda l: l.replace(">", "<") if ">" in l else l,
            lambda l: l.replace("<", ">") if "<" in l else l,
        ]
        
        mutation = random.choice(mutations)
        new_line = mutation(stripped)
        return indent + new_line
    
    def _simplify_line(self, line: str) -> str:
        """Simplify a line of code."""
        stripped = line.strip()
        indent = line[:len(line) - len(line.lstrip())]
        
        simplifications = {
            "return True": "return 1",
            "return False": "return 0",
            "x = x + 1": "x += 1",
            "x = x - 1": "x -= 1",
            "x = x * 1": "# no-op",
            "result.append(x)": "result += [x]",
        }
        
        for old, new in simplifications.items():
            if old in stripped:
                return indent + stripped.replace(old, new)
        
        return line
    
    def _generate_lines(self, n: int) -> list[str]:
        """Generate random valid Python lines."""
        templates = [
            "    pass",
            "    x = 0",
            "    result = []",
            "    return None",
            "    i += 1",
            "    if x: pass",
            "    for i in range(10): pass",
        ]
        return [random.choice(templates) for _ in range(n)]
    
    def _crossover(self, code1: str, code2: str) -> str:
        """Combine two code solutions."""
        lines1 = code1.splitlines()
        lines2 = code2.splitlines()
        
        # Take first half from code1, second half from code2
        split1 = len(lines1) // 2
        split2 = len(lines2) // 2
        
        child_lines = lines1[:split1] + lines2[split2:]
        
        # Ensure function structure
        if child_lines and not child_lines[0].strip().startswith("def "):
            if lines1 and lines1[0].strip().startswith("def "):
                child_lines.insert(0, lines1[0])
        
        return "\n".join(child_lines)
    
    def get_stats(self) -> dict:
        """Get evolution statistics."""
        if not self.history:
            return {"generations": 0}
        
        return {
            "generations": len(self.history),
            "best_fitness": self.best_ever.fitness if self.best_ever else 0,
            "best_code_lines": self.best_ever.lines if self.best_ever else 0,
            "population_size": len(self.population),
            "valid_organisms": sum(1 for o in self.population if o.is_valid),
        }
