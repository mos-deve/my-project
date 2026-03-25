"""
ANDROM - Adaptive Network of Deterministic Rule-based Operations and Mathematics
A self-improving rule-based system with thousands of mathematical units.
"""

from androm.unit import Unit, UnitType
from androm.network import Network
from androm.engine import RuleEngine
from androm.generator import CodeGenerator
from androm.optimizer import SelfOptimizer
from androm.brain import Brain

__version__ = "0.1.0"
__all__ = ["Unit", "UnitType", "Network", "RuleEngine", "CodeGenerator", "SelfOptimizer", "Brain"]
