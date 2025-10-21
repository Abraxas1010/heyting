# Unified Generative Ontology System - Quickstart

This notebook demonstrates the basic usage of the unified system.

## Installation & Setup

```python
# Add project to path
import sys
from pathlib import Path

project_root = Path("/content/drive/MyDrive/generative_ontology")
sys.path.insert(0, str(project_root))

# Import main package
import generative_ontology as go

# Show system info
go.system_info()
```

## Creating States

```python
# From vector
state = go.UnifiedState.from_vector([1, 2, 3, 4])
print(f"Created state: {state}")

# Zero state
zero = go.UnifiedState.zero(dimension=2)
print(f"Zero state: {zero}")

# Scalar state
scalar = go.UnifiedState.scalar(5.0, dimension=2)
print(f"Scalar state: {scalar}")
```

## Accessing Different Views

```python
# The same state can be viewed in three ways

# 1. Clifford algebra view
clifford_view = state.as_clifford()
print(f"Clifford: {clifford_view}")

# 2. Logic view
logic_view = state.as_logic()
print(f"Logic: {logic_view}")

# 3. Graph view
graph_view = state.as_graph()
print(f"Graph nodes: {graph_view.num_nodes}")
print(f"Graph edges: {graph_view.edge_index.shape[1]}")
```

## Performing Operations

### Clifford Operations

```python
from generative_ontology import get_clifford_bridge

bridge = get_clifford_bridge()

# Create basis vectors
e1 = go.UnifiedState.from_vector([0, 1, 0, 0])
e2 = go.UnifiedState.from_vector([0, 0, 1, 0])

# Wedge product (exterior product)
e12 = bridge.wedge_product(e1, e2)
print(f"e1 ∧ e2 = {e12.primary_data}")

# Inner product
inner = bridge.inner_product(e1, e1)
print(f"e1 · e1 = {inner.primary_data[0].item()}")
```

### Logic Operations

```python
from generative_ontology import get_logic_bridge

logic_bridge = get_logic_bridge()

# Meet operation (logical AND)
meet_result = logic_bridge.meet(e1, e2)
print(f"e1 ∧ e2 (logic) = {meet_result.primary_data if meet_result else None}")

# Join operation (logical OR)
join_result = logic_bridge.join(e1, e2)
print(f"e1 ∨ e2 = {join_result.primary_data}")

# Negation
not_e1 = logic_bridge.negate(e1)
print(f"¬e1 = {not_e1.primary_data}")
```

### Graph Operations

```python
from generative_ontology import get_graph_bridge

graph_bridge = get_graph_bridge()

# Convert to graph
graph = graph_bridge.state_to_graph(state)

# Query node information
node_info = graph_bridge.get_node_info(graph, 0, dimension=2)
print(f"Node 0: {node_info}")

# Get neighbors
neighbors = graph_bridge.get_neighbors(graph, 0, dimension=2)
print(f"Neighbors of node 0: {neighbors}")
```

## Verifying Round-Trip Consistency

```python
import torch

# Create original state
original = go.UnifiedState.from_vector([1, 2, 3, 4])

# Convert through Clifford and back
mv = get_clifford_bridge().state_to_clifford(original)
recovered = get_clifford_bridge().clifford_to_state(mv, dimension=2)

# Check error
error = torch.norm(original.primary_data - recovered.primary_data).item()
print(f"Round-trip error: {error:.2e}")
print(f"Within epsilon (10⁻¹⁰): {error < 1e-10}")
```

## Working with Different Dimensions

```python
# 1D (Heyting logic)
state_1d = go.UnifiedState.from_vector([1, 2])
print(f"1D logic type: {state_1d.logic_type}")

# 2D (Boolean logic)
state_2d = go.UnifiedState.from_vector([1, 2, 3, 4])
print(f"2D logic type: {state_2d.logic_type}")

# 3D (Boolean logic)
state_3d = go.UnifiedState.from_vector([1, 2, 3, 4, 5, 6, 7, 8])
print(f"3D logic type: {state_3d.logic_type}")
```

## Next Steps

- Explore the complete API in the documentation
- Run integration tests: `python tests/test_integration.py`
- Try building custom message passing layers
- Experiment with multi-modal learning

For more information, see the full documentation.
