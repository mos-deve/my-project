"""
CodeGenerator - Generates Python code from patterns and rules.
Uses the rule engine and network to produce executable code.
"""

from __future__ import annotations
import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CodeTemplate:
    """A code generation template."""
    name: str
    pattern: str  # Template pattern with {placeholders}
    category: str
    complexity: int = 1  # 1-10 scale
    
    def render(self, **kwargs) -> str:
        """Render template with given arguments."""
        try:
            return self.pattern.format(**kwargs)
        except KeyError as e:
            return f"# Missing template arg: {e}"


class CodeGenerator:
    """
    Generates Python code from problem descriptions.
    
    Uses pattern matching, templates, and learned code structures
    to produce working code solutions.
    """
    
    def __init__(self):
        self.templates: list[CodeTemplate] = []
        self.learned_patterns: dict[str, str] = {}
        self._init_default_templates()
    
    def _init_default_templates(self):
        """Initialize with basic code templates."""
        templates = [
            CodeTemplate(
                name="function_def",
                pattern="def {name}({params}):\n{body}",
                category="structure",
                complexity=1,
            ),
            CodeTemplate(
                name="loop_for",
                pattern="for {var} in {iterable}:\n{body}",
                category="control",
                complexity=1,
            ),
            CodeTemplate(
                name="loop_while",
                pattern="while {condition}:\n{body}",
                category="control",
                complexity=2,
            ),
            CodeTemplate(
                name="if_else",
                pattern="if {condition}:\n{then}\nelse:\n{else_body}",
                category="control",
                complexity=1,
            ),
            CodeTemplate(
                name="list_comp",
                pattern="[{expr} for {var} in {iterable}{filter}]",
                category="data",
                complexity=2,
            ),
            CodeTemplate(
                name="dict_comp",
                pattern="{{{key}: {value} for {var} in {iterable}}}",
                category="data",
                complexity=2,
            ),
            CodeTemplate(
                name="try_except",
                pattern="try:\n{body}\nexcept {exception} as {var}:\n{handler}",
                category="error",
                complexity=2,
            ),
            CodeTemplate(
                name="class_def",
                pattern="class {name}:\n{body}",
                category="structure",
                complexity=3,
            ),
            CodeTemplate(
                name="return_expr",
                pattern="return {expr}",
                category="control",
                complexity=1,
            ),
            CodeTemplate(
                name="assign",
                pattern="{name} = {value}",
                category="data",
                complexity=1,
            ),
        ]
        for t in templates:
            self.templates.append(t)
    
    def generate_function(self, name: str, params: list[str], 
                          logic: list[str], return_expr: str | None = None) -> str:
        """Generate a complete function."""
        param_str = ", ".join(params)
        
        # Build body
        body_lines = []
        for line in logic:
            body_lines.append(f"    {line}")
        
        if return_expr:
            body_lines.append(f"    return {return_expr}")
        
        body = "\n".join(body_lines) if body_lines else "    pass"
        
        return f"def {name}({param_str}):\n{body}"
    
    def generate_class(self, name: str, methods: list[dict]) -> str:
        """Generate a class with methods."""
        lines = [f"class {name}:"]
        
        if not methods:
            lines.append("    pass")
        else:
            for method in methods:
                m_name = method.get("name", "method")
                m_params = method.get("params", ["self"])
                m_body = method.get("body", ["pass"])
                m_return = method.get("return")
                
                param_str = ", ".join(m_params)
                body_lines = [f"        {line}" for line in m_body]
                if m_return:
                    body_lines.append(f"        return {m_return}")
                body = "\n".join(body_lines) if body_lines else "        pass"
                
                lines.append(f"\n    def {m_name}({param_str}):\n{body}")
        
        return "\n".join(lines)
    
    def analyze_code(self, code: str) -> dict:
        """Analyze code structure."""
        try:
            tree = ast.parse(code)
            
            functions = []
            classes = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        "name": node.name,
                        "args": [a.arg for a in node.args.args],
                        "lines": node.end_lineno - node.lineno + 1 if node.end_lineno else 1,
                    })
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        imports.extend(a.name for a in node.names)
                    else:
                        imports.append(node.module or "")
            
            return {
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "lines": len(code.splitlines()),
                "chars": len(code),
            }
        except SyntaxError as e:
            return {"error": str(e), "lines": len(code.splitlines()), "chars": len(code)}
    
    def learn_pattern(self, problem_type: str, solution: str):
        """Learn a code pattern for a problem type."""
        self.learned_patterns[problem_type] = solution
    
    def suggest_solution(self, problem_type: str) -> str | None:
        """Suggest a solution based on learned patterns."""
        return self.learned_patterns.get(problem_type)
    
    def generate_from_template(self, template_name: str, **kwargs) -> str:
        """Generate code from a named template."""
        for t in self.templates:
            if t.name == template_name:
                return t.render(**kwargs)
        return f"# Unknown template: {template_name}"
