"""Tests for ANDROM modules."""

import pytest
from androm.unit import Unit, UnitType
from androm.network import Network
from androm.engine import RuleEngine
from androm.generator import CodeGenerator
from androm.optimizer import SelfOptimizer, OptimizationResult
from androm.brain import Brain


class TestUnit:
    def test_math_unit(self):
        unit = Unit(id=0, unit_type=UnitType.MATH, inputs=[0, 1], weights=[1.0, 1.0], bias=0.0)
        result = unit.compute([0.5, 0.5])
        assert isinstance(result, float)
        assert -1 <= result <= 1  # tanh output range
    
    def test_logic_unit(self):
        unit = Unit(id=0, unit_type=UnitType.LOGIC, inputs=[0], weights=[1.0])
        assert unit.compute([1.0]) == 1.0
        assert unit.compute([-1.0]) == 0.0
    
    def test_compare_unit(self):
        unit = Unit(id=0, unit_type=UnitType.COMPARE, inputs=[0, 1], weights=[1.0, 1.0])
        assert unit.compute([1.0, 0.5]) == 1.0
        assert unit.compute([0.5, 1.0]) == 0.0
    
    def test_gate_unit(self):
        unit = Unit(id=0, unit_type=UnitType.GATE, inputs=[0, 1], weights=[1.0, 1.0])
        # Gate open (first input > 0.5)
        assert unit.compute([1.0, 0.8]) == 0.8
        # Gate closed
        assert unit.compute([0.0, 0.8]) == 0.0
    
    def test_memory_unit(self):
        unit = Unit(id=0, unit_type=UnitType.MEMORY, inputs=[0], weights=[1.0])
        r1 = unit.compute([1.0])
        r2 = unit.compute([0.0])
        # Memory should retain some value
        assert r2 != 0.0
    
    def test_serialization(self):
        unit = Unit(id=42, unit_type=UnitType.MATH, inputs=[0, 1], weights=[0.5, -0.5], bias=0.1)
        d = unit.to_dict()
        restored = Unit.from_dict(d)
        assert restored.id == 42
        assert restored.unit_type == UnitType.MATH
        assert restored.bias == 0.1


class TestNetwork:
    def test_basic_network(self):
        net = Network()
        i1 = net.add_input_unit()
        i2 = net.add_input_unit()
        h1 = net.add_unit(UnitType.MATH, inputs=[i1, i2], weights=[0.5, 0.5])
        o1 = net.add_output_unit([h1])
        
        outputs = net.propagate([1.0, 1.0])
        assert len(outputs) == 1
    
    def test_random_network(self):
        net = Network()
        for _ in range(10):
            net.add_input_unit()
        net.random_connect(100, connectivity=0.1)
        for _ in range(5):
            existing = list(net.units.keys())
            net.add_output_unit(existing[:3])
        
        assert net.size() == 115  # 10 input + 100 random + 5 output
    
    def test_serialization(self):
        net = Network()
        net.add_input_unit()
        net.random_connect(10, 0.2)
        
        d = net.to_dict()
        restored = Network.from_dict(d)
        assert restored.size() == net.size()


class TestRuleEngine:
    def test_basic_rules(self):
        engine = RuleEngine()
        engine.add_rule(
            "check_positive",
            lambda f: f.get("value", 0) > 0,
            lambda f: {"positive": True},
            priority=10,
        )
        engine.set_fact("value", 5)
        fired = engine.run()
        assert "check_positive" in fired
        assert engine.get_fact("positive") is True
    
    def test_rule_priority(self):
        engine = RuleEngine()
        order = []
        engine.add_rule("low", lambda f: True, lambda f: order.append("low"), priority=1)
        engine.add_rule("high", lambda f: True, lambda f: order.append("high"), priority=10)
        engine.run(max_fires=2)
        assert order[0] == "high"
    
    def test_no_fire(self):
        engine = RuleEngine()
        engine.add_rule("never", lambda f: False, lambda f: None)
        fired = engine.run()
        assert len(fired) == 0


class TestCodeGenerator:
    def test_generate_function(self):
        gen = CodeGenerator()
        code = gen.generate_function("add", ["a", "b"], [], "a + b")
        assert "def add(a, b):" in code
        assert "return a + b" in code
    
    def test_generate_class(self):
        gen = CodeGenerator()
        code = gen.generate_class("MyClass", [
            {"name": "init", "params": ["self"], "body": ["self.x = 0"]},
        ])
        assert "class MyClass:" in code
        assert "def init(self):" in code
    
    def test_analyze_code(self):
        gen = CodeGenerator()
        code = "def foo():\n    return 42"
        analysis = gen.analyze_code(code)
        assert len(analysis["functions"]) == 1
        assert analysis["functions"][0]["name"] == "foo"
    
    def test_template_rendering(self):
        gen = CodeGenerator()
        code = gen.generate_from_template("function_def", 
                                          name="test", params="x", body="    return x")
        assert "def test(x):" in code


class TestSelfOptimizer:
    def test_optimize_simple(self):
        opt = SelfOptimizer()
        code = """def foo():
    x = 1
    return None"""
        result = opt.optimize(code)
        assert result.is_valid
        # Should remove explicit return None
        assert "return None" not in result.optimized
    
    def test_optimize_comparisons(self):
        opt = SelfOptimizer()
        code = """def foo(x):
    if x == True:
        return True
    return False"""
        result = opt.optimize(code)
        assert result.is_valid
        # x == True should become just x
        assert "== True" not in result.optimized
    
    def test_self_optimization(self):
        opt = SelfOptimizer()
        result = opt.optimize_self()
        # Should be valid Python
        assert result.is_valid
    
    def test_stats(self):
        opt = SelfOptimizer()
        opt.optimize("def f():\n    return None")
        stats = opt.get_stats()
        assert stats["attempts"] >= 1


class TestBrain:
    def test_brain_build(self):
        brain = Brain()
        brain.build_network(100, 0.05)
        assert brain.network.size() >= 100
    
    def test_brain_think(self):
        brain = Brain()
        brain.build_network(50, 0.1)
        inputs = [0.5] * len(brain.network.input_ids)
        outputs = brain.think(inputs)
        assert len(outputs) == len(brain.network.output_ids)
    
    def test_brain_reason(self):
        brain = Brain()
        brain.engine.add_rule("test", lambda f: True, lambda f: {"result": "ok"})
        fired = brain.reason({"x": 1})
        assert "test" in fired
    
    def test_brain_solve(self):
        brain = Brain()
        brain.build_network(50, 0.1)
        code = brain.solve("test problem")
        assert "def solve" in code
    
    def test_brain_cycle(self):
        brain = Brain()
        brain.build_network(50, 0.1)
        result = brain.run_cycle()
        assert "cycle" in result
        assert "outputs" in result
    
    def test_brain_optimize(self):
        brain = Brain()
        results = brain.optimize_self()
        assert isinstance(results, dict)
    
    def test_brain_status(self):
        brain = Brain()
        brain.build_network(100, 0.05)
        status = brain.status()
        assert status["units"] >= 100
        assert status["cycles"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
