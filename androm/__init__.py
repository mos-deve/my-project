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
from androm.model import Model
from androm.learner import Learner
from androm.evolver import Evolver, CodeOrganism
from androm.scenarios import Scenario, ScenarioLibrary

__version__ = "0.2.0"
__all__ = [
    "Unit", "UnitType", "Network", "RuleEngine", "CodeGenerator", 
    "SelfOptimizer", "Brain", "Model", "Learner", "Evolver", 
    "CodeOrganism", "Scenario", "ScenarioLibrary"
]
