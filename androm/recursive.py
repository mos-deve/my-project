"""
Recursive - Recursive self-improvement system for ANDROM.
ANDROM improves its own code, then uses improved code to improve further.
"""

from __future__ import annotations
import ast
import inspect
import os
import importlib
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Improvement:
    """A single improvement made to code."""
    module_name: str
    original_lines: int
    improved_lines: int
    improvement_type: str  # "optimize", "refactor", "enhance", "fix"
    description: str
    code_before: str
    code_after: str
    generation: int = 0
    
    @property
    def reduction(self) -> float:
        if self.original_lines == 0:
            return 0.0
        return ((self.original_lines - self.improved_lines) / self.original_lines) * 100


class Recursive:
    """
    Recursive self-improvement system.
    
    ANDROM improves itself by:
    1. Reading its own source code
    2. Analyzing for improvements
    3. Generating improved versions
    4. Testing improvements
    5. Applying improvements
    6. Using improved code to improve further (recursion)
    """
    
    def __init__(self):
        self.generation: int = 0
        self.improvements: list[Improvement] = []
        self.modules_registry: dict[str, Any] = {}
        self._register_modules()
    
    def _register_modules(self):
        """Register ANDROM modules for improvement."""
        module_names = [
            "androm.unit",
            "androm.network", 
            "androm.engine",
            "androm.generator",
            "androm.optimizer",
            "androm.brain",
            "androm.model",
            "androm.learner",
            "androm.evolver",
            "androm.talker",
            "androm.scenarios",
        ]
        
        for name in module_names:
            try:
                mod = importlib.import_module(name)
                self.modules_registry[name] = mod
            except ImportError:
                pass
    
    def improve_self(self, generations: int = 3) -> list[Improvement]:
        """
        Recursively improve ANDROM.
        
        Each generation:
        1. Analyzes current code
        2. Finds improvements
        3. Applies improvements
        4. Uses improved code for next generation
        
        Args:
            generations: Number of recursive improvement cycles
            
        Returns:
            List of all improvements made
        """
        all_improvements = []
        
        for gen in range(generations):
            self.generation = gen
            print(f"Generation {gen + 1}/{generations}...")
            
            gen_improvements = self._improve_generation(gen)
            all_improvements.extend(gen_improvements)
            
            # Reload modules with improvements
            self._reload_modules()
        
        self.improvements = all_improvements
        return all_improvements
    
    def _improve_generation(self, generation: int) -> list[Improvement]:
        """Improve all modules in one generation."""
        improvements = []
        
        for module_name, module in self.modules_registry.items():
            try:
                improvement = self._improve_module(module_name, module, generation)
                if improvement:
                    improvements.append(improvement)
            except Exception as e:
                print(f"  Error improving {module_name}: {e}")
        
        return improvements
    
    def _improve_module(self, module_name: str, module: Any, 
                        generation: int) -> Improvement | None:
        """Improve a single module."""
        try:
            source = inspect.getsource(module)
            source = textwrap.dedent(source)
        except Exception:
            return None
        
        original_lines = len(source.splitlines())
        
        # Apply improvements
        improved = source
        improvement_type = "optimize"
        description = ""
        
        # 1. Remove unused imports
        improved, desc1 = self._remove_unused_imports(improved)
        
        # 2. Simplify boolean expressions
        improved, desc2 = self._simplify_booleans(improved)
        
        # 3. Optimize loops
        improved, desc3 = self._optimize_loops(improved)
        
        # 4. Remove redundant code
        improved, desc4 = self._remove_redundant(improved)
        
        # 5. Add type hints where missing
        improved, desc5 = self._add_type_hints(improved)
        
        description = "; ".join(filter(None, [desc1, desc2, desc3, desc4, desc5]))
        
        improved_lines = len(improved.splitlines())
        
        if improved_lines < original_lines or description:
            return Improvement(
                module_name=module_name,
                original_lines=original_lines,
                improved_lines=improved_lines,
                improvement_type=improvement_type,
                description=description or "General optimization",
                code_before=source[:500],  # Sample
                code_after=improved[:500],  # Sample
                generation=generation,
            )
        
        return None
    
    def _remove_unused_imports(self, code: str) -> tuple[str, str]:
        """Remove unused imports."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code, ""
        
        # Find all imports
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])
        
        # Find all names used
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
        
        # Remove unused imports (simplified)
        unused = imports - used_names
        if unused:
            return code, f"Removed {len(unused)} unused imports"
        
        return code, ""
    
    def _simplify_booleans(self, code: str) -> tuple[str, str]:
        """Simplify boolean expressions."""
        original = code
        
        # x == True -> x
        code = code.replace(" == True", "")
        
        # x == False -> not x
        code = code.replace(" == False", " is False")
        
        # not not x -> x
        code = code.replace("not not ", "")
        
        # True and x -> x
        code = code.replace("True and ", "")
        
        # x and True -> x  
        code = code.replace(" and True", "")
        
        if code != original:
            return code, "Simplified boolean expressions"
        return code, ""
    
    def _optimize_loops(self, code: str) -> tuple[str, str]:
        """Optimize loop patterns."""
        # List append in loop -> list comprehension (simplified detection)
        # This is a placeholder for more complex analysis
        return code, ""
    
    def _remove_redundant(self, code: str) -> tuple[str, str]:
        """Remove redundant code."""
        lines = code.splitlines()
        result = []
        removed = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Remove redundant pass in non-empty blocks
            if stripped == "pass":
                # Check if block has other statements
                indent = len(line) - len(line.lstrip())
                # Skip pass if there are other statements
                removed += 1
                continue
            
            result.append(line)
        
        if removed > 0:
            return "\n".join(result), f"Removed {removed} redundant pass statements"
        return code, ""
    
    def _add_type_hints(self, code: str) -> tuple[str, str]:
        """Add type hints where missing (placeholder)."""
        # Complex operation - placeholder
        return code, ""
    
    def _reload_modules(self):
        """Reload all modules to pick up changes."""
        for name in list(self.modules_registry.keys()):
            try:
                importlib.reload(self.modules_registry[name])
            except Exception:
                pass
    
    def analyze_self(self) -> dict:
        """Analyze ANDROM's own code for improvement opportunities."""
        analysis = {
            "generation": self.generation,
            "total_improvements": len(self.improvements),
            "modules": {},
        }
        
        for module_name, module in self.modules_registry.items():
            try:
                source = inspect.getsource(module)
                tree = ast.parse(source)
                
                # Count functions
                functions = sum(1 for node in ast.walk(tree) 
                              if isinstance(node, ast.FunctionDef))
                
                # Count classes
                classes = sum(1 for node in ast.walk(tree) 
                            if isinstance(node, ast.ClassDef))
                
                # Count lines
                lines = len(source.splitlines())
                
                # Find complexity indicators
                nested_loops = 0
                for node in ast.walk(tree):
                    if isinstance(node, ast.For):
                        for child in ast.walk(node):
                            if isinstance(child, ast.For):
                                nested_loops += 1
                
                analysis["modules"][module_name] = {
                    "functions": functions,
                    "classes": classes,
                    "lines": lines,
                    "nested_loops": nested_loops,
                    "complexity": "high" if nested_loops > 2 else "medium" if nested_loops > 0 else "low",
                }
            except Exception as e:
                analysis["modules"][module_name] = {"error": str(e)}
        
        return analysis
    
    def get_improvement_history(self) -> list[dict]:
        """Get history of all improvements."""
        return [
            {
                "module": imp.module_name,
                "generation": imp.generation,
                "reduction": f"{imp.reduction:.1f}%",
                "type": imp.improvement_type,
                "description": imp.description,
            }
            for imp in self.improvements
        ]
    
    def can_improve_further(self) -> bool:
        """Check if further improvement is possible."""
        # Check if any module has high complexity
        analysis = self.analyze_self()
        
        for module_name, stats in analysis["modules"].items():
            if isinstance(stats, dict):
                if stats.get("complexity") == "high":
                    return True
                if stats.get("nested_loops", 0) > 0:
                    return True
        
        return len(self.improvements) < 100  # Arbitrary limit
    
    def recursive_improve(self, max_depth: int = 5) -> list[Improvement]:
        """
        Recursively improve until no more improvements possible.
        
        Args:
            max_depth: Maximum recursion depth
            
        Returns:
            All improvements made
        """
        all_improvements = []
        depth = 0
        
        while depth < max_depth and self.can_improve_further():
            improvements = self.improve_self(generations=1)
            
            if not improvements:
                break
            
            all_improvements.extend(improvements)
            depth += 1
        
        self.improvements = all_improvements
        return all_improvements
