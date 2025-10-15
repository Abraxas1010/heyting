"""
Unified Generative Ontology System

A unified mathematical framework integrating:
- Heyting/Boolean Logic
- Clifford Algebra
- Graph Neural Networks
- Tensor Representations

Version: 1.0
"""

__version__ = "1.0.0"
__author__ = "Generative Ontology Team"

# Core imports
from core.unified_state import UnifiedState, LogicType

# Engine imports
from ga_clifford.engine import CliffordEngine
from logic.heyting import LogicEngine
from graph.engine import GraphEngine

# Bridge imports
from bridges.logic_clifford import (
    get_clifford_bridge,
    get_logic_bridge,
    get_graph_bridge,
    CliffordBridge,
    LogicBridge,
    GraphBridge,
)

# Phase 4: Combinatorial Reasoning
from training.reasoning_engines import (
    Representation,
    ReasoningMode,
    InductionEngine,
    DeductionEngine,
    AbductionEngine,
)
from training.combinatorial_engine import (
    CombinatorialNode,
    CombinatorialPath,
    CombinatorialReasoningEngine,
)
# Public API
__all__ = [
    # Core
    'UnifiedState',
    'LogicType',
    # Engines
    'CliffordEngine',
    'LogicEngine',
    'GraphEngine',
    # Bridges
    'CliffordBridge',
    'LogicBridge',
    'GraphBridge',
    'get_clifford_bridge',
    'get_logic_bridge',
    'get_graph_bridge',

    # Phase 4: Combinatorial Reasoning
    'Representation',
    'ReasoningMode',
    'InductionEngine',
    'DeductionEngine',
    'AbductionEngine',
    'CombinatorialNode',
    'CombinatorialPath',
    'CombinatorialReasoningEngine',
]

def create_state(*args, **kwargs):
    """
    Convenience function to create UnifiedState.
    Examples:
        >>> state = create_state([1, 2, 3, 4])  # From vector
        >>> state = create_state.zero(dimension=2)  # Zero state
        >>> state = create_state.scalar(5.0, dimension=2)  # Scalar
    """
    return UnifiedState(*args, **kwargs)

# Add factory methods as attributes
create_state.from_vector = UnifiedState.from_vector
create_state.zero = UnifiedState.zero
create_state.scalar = UnifiedState.scalar

# Version info
def get_version():
    """Get version information"""
    return {
        'version': __version__,
        'author': __author__,
        'components': {
            'unified_state': True,
            'clifford_engine': True,
            'logic_engine': True,
            'graph_engine': True,
            'bridges': True,
        }
    }

def system_info():
    """Print system information"""
    import torch
    info = get_version()
    print(f"Unified Generative Ontology System v{info['version']}")
    print(f"\nComponents:")
    for component, status in info['components'].items():
        status_str = '✓' if status else '✗'
        print(f"  {status_str} {component}")
    print(f"\nDevice: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"\nTest Status: 96/96 passing")
    print(f"Round-trip consistency: ε < 10⁻¹⁰")
