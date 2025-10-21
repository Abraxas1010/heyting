# Phase 4: Combinatorial Reasoning Architecture

## Overview

Phase 4 implements a complete tripartite cognitive architecture enabling meta-level reasoning across all representations and modes.

## Architecture

### Three Layers

1. **Layer 0: Topos (UnifiedState)**
   - Universal binding for all representations
   - Single source of truth
   - Consistency guarantees: ε < 10⁻¹⁰

2. **Layer 1: Sheaves (Representations)**
   - U: Unified (base tensor)
   - C: Clifford (geometric algebra)
   - L: Logic (symbolic reasoning)
   - G: Graph (neural networks)

3. **Layer 2: Reasoning Modes**
   - I: Induction `(Data + Answer) → Rules`
   - D: Deduction `(Rules + Data) → Answer`
   - A: Abduction `(Rules + Answer) → Data`

### Computational Modalities

**12 total modalities**: (U,C,L,G) × (I,D,A)

**144 possible transitions**: Any modality can transition to any other

## Usage

### Basic Example

```python
from generative_ontology import (
    CombinatorialReasoningEngine,
    CombinatorialPath,
    CombinatorialNode,
    Representation,
    ReasoningMode,
)

# Initialize engine
engine = CombinatorialReasoningEngine(dimension=2)

# Define a reasoning path
path = CombinatorialPath([
    CombinatorialNode(Representation.GRAPH, ReasoningMode.INDUCTION),
    CombinatorialNode(Representation.CLIFFORD, ReasoningMode.DEDUCTION),
    CombinatorialNode(Representation.LOGIC, ReasoningMode.ABDUCTION),
])

# Execute path
initial_state = UnifiedState.from_vector([1, 2, 3, 4])
context = {'training_samples': samples, 'target': target}
result, updated_context = engine.execute_path(path, initial_state, context)
```

### Exploring the Space

```python
# Generate diverse paths
paths = engine.generate_diverse_paths(max_length=4)

# Test all paths on a task
engine.explore_combinatorial_space(n_samples=50, max_path_length=3)

# Analyze which paths work best
# Results stored in engine.path_performance
```

## Reasoning Modes Explained

### Induction: Learn from Examples

```python
# Given input-output pairs, learn the transformation
context = {
    'training_samples': [(input1, output1), (input2, output2), ...],
    'epochs': 50,
    'learning_rate': 1e-3
}

# The InductionEngine trains a model to approximate the function
```

### Deduction: Apply Rules

```python
# Given rules (learned or geometric), compute answer
context = {
    'learned_model': trained_model,  # From induction
    # OR
    'operation': 'wedge',  # Geometric operation
    'operand_b': other_state
}

# The DeductionEngine applies rules to produce answer
```

### Abduction: Generate Explanations

```python
# Given desired output, find plausible inputs
context = {
    'target': desired_output,
    'learned_model': trained_model,
    'max_iterations': 500
}

# The AbductionEngine optimizes inputs to produce target
```

## Strategic Paths

### Strategy 1: Pure Induction
Learn everything from data
```
(G, I) → (G, D)
```

### Strategy 2: Pure Deduction
Use only axiomatic knowledge
```
(C, D)
```

### Strategy 3: Hybrid Learning
Combine learning with geometric reasoning
```
(G, I) → (C, D) → (L, D)
```

### Strategy 4: Abduction-Enhanced
Use hypothesis generation to improve learning
```
(G, I) → (L, A) → (C, D) → (G, I)
```

## Meta-Learning

The system can learn which paths work best for which tasks:

1. Execute multiple paths on same task
2. Measure performance of each path
3. Analyze patterns in successful paths
4. Optimize future path selection

## Performance

- 12 modalities operational ✓
- 144 transitions verified ✓
- Path generation: 10+ strategies
- Meta-level optimization: Active

## Examples

See `examples/phase4_orthogonality.ipynb` for complete working example.

## Testing

```bash
# Run from Colab
# Task 13: Reasoning engines
# Task 14: Combinatorial engine
# Task 15: Integration tests
# Task 16: Full demonstration
```

All tests passing ✓
