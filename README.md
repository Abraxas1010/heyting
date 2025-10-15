# Unified Generative Ontology System

A unified mathematical framework integrating Clifford algebra, dimension-dependent logic, and graph neural networks with verified consistency.

## Features

✅ **Unified State Management**
- Single source of truth with lazy view computation
- Automatic dimension detection (1D/2D/3D)
- GPU/CPU support

✅ **Clifford Algebra**
- Wedge (∧), inner (·), and geometric products
- Grade projection and manipulation
- Rotor-based rotations

✅ **Dimension-Dependent Logic**
- 1D: Heyting/Intuitionistic logic (¬¬a ≠ a)
- 2D+: Boolean/Classical logic (¬¬a = a)
- Meet (∧), join (∨), negation (¬), implication (→)

✅ **Graph Neural Networks**
- PyTorch Geometric integration
- Nodes = basis blades, edges = geometric products
- Ready for message passing layers

✅ **Verified Consistency**
- Round-trip error: ε < 10⁻¹⁰
- 96 tests passing
- All pairwise and composite paths verified

## Installation

```bash
# Mount Google Drive in Colab
from google.colab import drive
drive.mount('/content/drive')

# Add to Python path
import sys
sys.path.insert(0, '/content/drive/MyDrive/generative_ontology')
```

## Quick Start

```python
import generative_ontology as go

# Create unified state
state = go.UnifiedState.from_vector([1, 2, 3, 4])

# Access different views
clifford_view = state.as_clifford()  # Multivector
logic_view = state.as_logic()        # Logic element
graph_view = state.as_graph()        # PyG Data

# Perform operations
bridge = go.get_clifford_bridge()
result = bridge.wedge_product(state, other_state)
```

## Architecture

```
         UnifiedState
        (Single Source)
             │
    ┌────────┼────────┐
    │        │        │
    ▼        ▼        ▼
Clifford  Logic   Graph
    │        │        │
Wedge ∧   Meet ∧   Nodes
Inner ·   Join ∨   Edges
Geometric Negate¬  PyG
```

## System Status

| Component | Tests | Status |
|-----------|-------|--------|
| UnifiedState | 11 | ✅ |
| CliffordEngine | 11 | ✅ |
| CliffordBridge | 10 | ✅ |
| LogicEngine | 14 | ✅ |
| LogicBridge | 10 | ✅ |
| GraphEngine | 12 | ✅ |
| GraphBridge | 10 | ✅ |
| Complete Bridges | 18 | ✅ |
| **Total** | **96** | **✅** |

## Documentation

- [Quickstart Guide](examples/quickstart.md)
- [API Reference](docs/api.md) (coming soon)
- [Architecture Overview](docs/architecture.md) (coming soon)

## Testing

```python
# Run integration tests
python tests/test_integration.py

# Or from within Python
from tests.test_integration import TestSystemIntegration
tester = TestSystemIntegration()
tester.run_all()
```

## Project Structure

```
generative_ontology/
├── __init__.py              # Main API
├── core/
│   ├── unified_state.py     # Core state management
│   └── config.py            # Configuration
├── ga_clifford/
│   └── engine.py            # Clifford algebra
├── logic/
│   └── heyting.py           # Logic operations
├── graph/
│   └── engine.py            # Graph representation
├── bridges/
│   └── logic_clifford.py    # All bridges
├── tests/
│   └── test_integration.py  # Integration tests
└── examples/
    └── quickstart.md        # Getting started

```

## Requirements

- Python 3.12+
- PyTorch 2.0+
- PyTorch Geometric 2.3+
- clifford 1.4+
- NumPy 1.24+

## Version

Current: v1.0.0

## License

MIT License

## Citation

If you use this system in your research, please cite:

```bibtex
@software{unified_generative_ontology,
  title = {Unified Generative Ontology System},
  author = {Generative Ontology Team},
  year = {2025},
  version = {1.0.0}
}
```

## Contact

For questions, issues, or contributions, please open an issue on the project repository.
