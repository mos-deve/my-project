"""
RuleEngine - Deterministic rule-based reasoning system.
Pattern matching, inference chains, and decision making.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Rule:
    """A single rule: IF condition THEN action."""
    name: str
    condition: Callable[[dict], bool]
    action: Callable[[dict], Any]
    priority: int = 0
    fired_count: int = 0
    
    def matches(self, facts: dict) -> bool:
        """Check if rule condition is satisfied."""
        try:
            return self.condition(facts)
        except Exception:
            return False
    
    def fire(self, facts: dict) -> Any:
        """Execute rule action."""
        self.fired_count += 1
        return self.action(facts)


@dataclass
class RuleEngine:
    """
    Forward-chaining rule engine.
    
    Processes rules against a working memory of facts.
    Rules fire when their conditions match, modifying facts.
    """
    
    rules: list[Rule] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)
    max_iterations: int = 1000
    
    def add_rule(self, name: str, condition: Callable, action: Callable,
                 priority: int = 0):
        """Add a rule to the engine."""
        rule = Rule(name=name, condition=condition, action=action, priority=priority)
        self.rules.append(rule)
        self.rules.sort(key=lambda r: -r.priority)  # Higher priority first
    
    def set_fact(self, key: str, value: Any):
        """Set a fact in working memory."""
        self.facts[key] = value
    
    def get_fact(self, key: str, default: Any = None) -> Any:
        """Get a fact from working memory."""
        return self.facts.get(key, default)
    
    def run(self, max_fires: int | None = None, once: bool = False) -> list[str]:
        """
        Run the rule engine until no rules match or max_fires reached.
        
        Args:
            max_fires: Maximum number of rule firings
            once: If True, each rule fires at most once per run
            
        Returns:
            List of fired rule names (trace)
        """
        self.trace = []
        fires = 0
        max_f = max_fires or self.max_iterations
        fired_this_run: set[str] = set()
        
        while fires < max_f:
            fired = False
            for rule in self.rules:
                # Skip if once mode and already fired
                if once and rule.name in fired_this_run:
                    continue
                    
                if rule.matches(self.facts):
                    old_facts = self.facts.copy()
                    result = rule.fire(self.facts)
                    
                    # Apply result if it's a dict
                    if isinstance(result, dict):
                        self.facts.update(result)
                    
                    # Only count as fired if facts actually changed
                    if self.facts != old_facts:
                        self.trace.append(rule.name)
                        fired_this_run.add(rule.name)
                        fires += 1
                        fired = True
                        break  # Re-evaluate from highest priority
            
            if not fired:
                break
        
        return self.trace
    
    def infer(self, query: str) -> Any:
        """Query a fact after running inference."""
        self.run()
        return self.facts.get(query)
    
    def clear(self):
        """Clear facts and trace."""
        self.facts.clear()
        self.trace.clear()
    
    def rule_stats(self) -> list[dict]:
        """Get statistics about rule firing."""
        return [
            {"name": r.name, "fired": r.fired_count}
            for r in self.rules
        ]
