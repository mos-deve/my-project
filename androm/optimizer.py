"""
SelfOptimizer - The self-improvement engine.
Reads, analyzes, and optimizes code - including its own source.
"""

from __future__ import annotations
import ast
import inspect
import textwrap
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable
from androm.generator import CodeGenerator


@dataclass
class OptimizationResult:
    """Result of an optimization attempt."""
    original: str
    optimized: str
    original_lines: int
    optimized_lines: int
    reduction_percent: float
    is_valid: bool
    error: str | None = None
    
    @property
    def improved(self) -> bool:
        return self.is_valid and self.optimized_lines < self.original_lines


class SelfOptimizer:
    """
    Optimizes code by applying transformation rules.
    Can read its own source and generate shorter equivalent versions.
    """
    
    def __init__(self, generator: CodeGenerator | None = None):
        self.generator = generator or CodeGenerator()
        self.transformations: list[Callable[[str], str]] = []
        self.history: list[OptimizationResult] = []
        self._init_transformations()
    
    def _init_transformations(self):
        """Initialize code transformation rules."""
        self.transformations = [
            self._remove_pass_statements,
            self._simplify_returns,
            self._combine_assignments,
            self._remove_redundant_else,
            self._simplify_comparisons,
            self._use_list_comprehensions,
            self._remove_redundant_parens,
            self._simplify_bool_expressions,
            self._shorten_common_patterns,
            self._remove_blank_lines,
        ]
    
    def optimize(self, code: str, max_iterations: int = 10) -> OptimizationResult:
        """
        Iteratively optimize code until no more improvements found.
        """
        original = code
        current = code
        
        for _ in range(max_iterations):
            previous = current
            for transform in self.transformations:
                current = transform(current)
            
            if current == previous:
                break
        
        # Validate optimized code
        is_valid, error = self._validate_code(current)
        
        original_lines = len(original.strip().splitlines())
        optimized_lines = len(current.strip().splitlines())
        reduction = ((original_lines - optimized_lines) / original_lines * 100) if original_lines > 0 else 0
        
        result = OptimizationResult(
            original=original,
            optimized=current,
            original_lines=original_lines,
            optimized_lines=optimized_lines,
            reduction_percent=reduction,
            is_valid=is_valid,
            error=error,
        )
        self.history.append(result)
        return result
    
    def optimize_self(self) -> OptimizationResult:
        """
        Optimize this module's own source code.
        The key self-improvement function.
        """
        # Read own source
        source = inspect.getsource(SelfOptimizer)
        source = textwrap.dedent(source)
        return self.optimize(source)
    
    def optimize_module(self, module: Any) -> OptimizationResult:
        """Optimize an entire module's source."""
        source = inspect.getsource(module)
        source = textwrap.dedent(source)
        return self.optimize(source)
    
    def _validate_code(self, code: str) -> tuple[bool, str | None]:
        """Check if code is syntactically valid Python."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)
    
    # === Transformation Rules ===
    
    def _remove_pass_statements(self, code: str) -> str:
        """Remove unnecessary pass statements."""
        lines = code.splitlines()
        result = []
        for line in lines:
            stripped = line.strip()
            if stripped == "pass":
                # Check if pass is needed (empty block)
                indent = len(line) - len(line.lstrip())
                # Skip if next line has same or less indent
                continue
            result.append(line)
        return "\n".join(result)
    
    def _simplify_returns(self, code: str) -> str:
        """Simplify return statements."""
        # return x if condition else y -> could be ternary but harder to read
        # Simplify: return None -> removed (implicit)
        lines = code.splitlines()
        result = []
        for line in lines:
            stripped = line.strip()
            if stripped == "return None":
                # Remove explicit None return (implicit at end of function)
                continue
            result.append(line)
        return "\n".join(result)
    
    def _combine_assignments(self, code: str) -> str:
        """Combine consecutive assignments where possible."""
        # x = 0; y = 0 -> x = y = 0 (but Python doesn't support inline)
        # Skip for now
        return code
    
    def _remove_redundant_else(self, code: str) -> str:
        """Remove else after return/break/continue."""
        lines = code.splitlines()
        result = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check if previous line was return/break/continue
            if result:
                prev = result[-1].strip()
                if stripped.startswith("else:") and prev in ("return", "break", "continue", "pass"):
                    # Skip the else:
                    i += 1
                    # De-indent the else body
                    if i < len(lines):
                        else_line = lines[i]
                        indent = len(line) - len(line.lstrip())
                        else_indent = len(else_line) - len(else_line.lstrip())
                        if else_indent > indent:
                            deindent = " " * indent
                            result.append(deindent + else_line.strip())
                            i += 1
                            while i < len(lines):
                                curr_indent = len(lines[i]) - len(lines[i].lstrip())
                                if curr_indent > indent:
                                    result.append(deindent + lines[i].strip())
                                    i += 1
                                else:
                                    break
                            continue
            
            result.append(line)
            i += 1
        
        return "\n".join(result)
    
    def _simplify_comparisons(self, code: str) -> str:
        """Simplify comparison expressions."""
        # x == True -> x
        code = code.replace("== True", "")
        # x == False -> not x
        code = code.replace("== False", " is False")
        # x != None -> x is not None
        code = code.replace("!= None", " is not None")
        # x == None -> x is None
        code = code.replace("== None", " is None")
        return code
    
    def _use_list_comprehensions(self, code: str) -> str:
        """Convert simple loops to comprehensions where possible."""
        # This is complex AST transformation, skip for v1
        return code
    
    def _remove_redundant_parens(self, code: str) -> str:
        """Remove unnecessary parentheses."""
        # Simple case: ((x)) -> (x)
        # Dangerous to do with regex on code, skip for now
        return code
    
    def _simplify_bool_expressions(self, code: str) -> str:
        """Simplify boolean expressions."""
        # not not x -> x (but only at expression level)
        code = code.replace("not not ", "")
        # True and x -> x
        code = code.replace("True and ", "")
        # x and True -> x
        code = code.replace(" and True", "")
        return code
    
    def _shorten_common_patterns(self, code: str) -> str:
        """Shorten common code patterns."""
        # len(x) == 0 -> not x
        code = code.replace("len(", "len(")  # Placeholder
        # x == [] -> not x
        # These are risky, skip for v1
        return code
    
    def _remove_blank_lines(self, code: str) -> str:
        """Remove excessive blank lines (keep max 1)."""
        lines = code.splitlines()
        result = []
        blank_count = 0
        for line in lines:
            if line.strip() == "":
                blank_count += 1
                if blank_count <= 1:
                    result.append(line)
            else:
                blank_count = 0
                result.append(line)
        return "\n".join(result)
    
    def get_stats(self) -> dict:
        """Get optimization statistics."""
        if not self.history:
            return {"attempts": 0}
        
        successful = [r for r in self.history if r.improved]
        return {
            "attempts": len(self.history),
            "successful": len(successful),
            "avg_reduction": sum(r.reduction_percent for r in successful) / len(successful) if successful else 0,
            "best_reduction": max((r.reduction_percent for r in successful), default=0),
        }
