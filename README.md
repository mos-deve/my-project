# ANDROM

**Adaptive Network of Deterministic Rule-based Operations and Mathematics**

A self-improving rule-based system with thousands of mathematical units. No neural networks - pure deterministic + probabilistic logic.

## What is it?

ANDROM is a brain-like system made of thousands of small mathematical parts that:
- Process data through interconnected computational units
- Reason using a rule engine
- Generate code from patterns
- **Optimize itself** - can shorten its own source code

## Architecture

```
Brain (orchestrator)
├── Network (1000+ units)
│   ├── Unit: MATH, LOGIC, COMPARE, AGGREGATE, TRANSFORM, GATE, MEMORY, PROBABILISTIC
│   └── Signal propagation through connections
├── RuleEngine (forward-chaining reasoning)
├── CodeGenerator (template-based code generation)
└── SelfOptimizer (self-improvement engine)
```

## Modules

| Module | Purpose |
|--------|---------|
| `unit.py` | 8 types of computational units - neuron-like parts that do math |
| `network.py` | Connects units, propagates signals through the graph |
| `engine.py` | Deterministic rule-based reasoning (IF condition THEN action) |
| `generator.py` | Generates Python code from templates and patterns |
| `optimizer.py` | Reads code, applies transformations, shortens it |
| `brain.py` | Orchestrates all modules into a thinking system |
| `cli.py` | Command-line interface |

## Installation

```bash
pip install -e .
```

## Usage

### CLI

```bash
# Run 10 thinking cycles with 1000 units
python -m androm.cli --units 1000 --cycles 10

# Run self-optimization
python -m androm.cli --optimize

# Show brain status
python -m androm.cli --status
```

### Python API

```python
from androm import Brain

# Create brain with 1000 units
brain = Brain()
brain.build_network(num_units=1000, connectivity=0.05)

# Run thinking cycles
result = brain.run_cycle()
print(result)

# Self-optimize
results = brain.optimize_self()
for module, stats in results.items():
    print(f"{module}: {stats['original_lines']} → {stats['optimized_lines']} lines")
```

## Unit Types

| Type | Function |
|------|----------|
| MATH | Weighted sum + tanh activation |
| LOGIC | Binary step (0 or 1) |
| COMPARE | Compares two inputs |
| AGGREGATE | Mean of inputs |
| TRANSFORM | Sigmoid transformation |
| GATE | Conditional gating (first input controls second) |
| MEMORY | Stateful - remembers previous values |
| PROBABILISTIC | Random decisions based on weighted sum |

## Self-Optimization

The optimizer can read and shorten its own source code:

```
Original (258 lines)
Optimized (256 lines)
Reduction: 0.8%
```

It applies transformations like:
- Remove unnecessary `return None`
- Simplify comparisons (`== True` → just the variable)
- Remove redundant else after return
- Remove excessive blank lines

## Testing

```bash
python tests/test_androm.py
```

## License

MIT
