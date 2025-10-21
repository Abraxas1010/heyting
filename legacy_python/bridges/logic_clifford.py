"""
logic_clifford.py

Bidirectional representation converters

Bridge between UnifiedState and CliffordEngine for seamless conversion.
"""

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.unified_state import UnifiedState
    from ga_clifford.engine import CliffordEngine


class CliffordBridge:
    """Bridge between UnifiedState and CliffordEngine."""

    def __init__(self):
        self._engines = {}

    def get_engine(self, dimension: int):
        """Get or create CliffordEngine for dimension."""
        if dimension not in self._engines:
            from ga_clifford.engine import CliffordEngine
            self._engines[dimension] = CliffordEngine(dimension)
        return self._engines[dimension]

    def state_to_clifford(self, state):
        """Convert UnifiedState to Clifford multivector."""
        engine = self.get_engine(state.dimension)
        return engine.tensor_to_multivector(state.primary_data)

    def clifford_to_state(self, multivector, dimension: int):
        """Convert Clifford multivector to UnifiedState."""
        from core.unified_state import UnifiedState
        engine = self.get_engine(dimension)
        tensor_data = engine.multivector_to_tensor(multivector)
        return UnifiedState(tensor_data, dimension)

    def wedge_product(self, state_a, state_b):
        """Compute wedge product."""
        from core.unified_state import UnifiedState
        if state_a.dimension != state_b.dimension:
            raise ValueError(f"Dimension mismatch")
        engine = self.get_engine(state_a.dimension)
        result_tensor = engine.wedge_product(state_a.primary_data, state_b.primary_data)
        return UnifiedState(result_tensor, state_a.dimension)

    def inner_product(self, state_a, state_b):
        """Compute inner product."""
        from core.unified_state import UnifiedState
        if state_a.dimension != state_b.dimension:
            raise ValueError(f"Dimension mismatch")
        engine = self.get_engine(state_a.dimension)
        result_tensor = engine.inner_product(state_a.primary_data, state_b.primary_data)
        return UnifiedState(result_tensor, state_a.dimension)

    def geometric_product(self, state_a, state_b):
        """Compute geometric product."""
        from core.unified_state import UnifiedState
        if state_a.dimension != state_b.dimension:
            raise ValueError(f"Dimension mismatch")
        engine = self.get_engine(state_a.dimension)
        result_tensor = engine.geometric_product(state_a.primary_data, state_b.primary_data)
        return UnifiedState(result_tensor, state_a.dimension)


_bridge = None

def get_clifford_bridge():
    """Get global CliffordBridge instance"""
    global _bridge
    if _bridge is None:
        _bridge = CliffordBridge()
    return _bridge


# ============================================================================
# LOGIC BRIDGE
# ============================================================================

class LogicBridge:
    """
    Bridge between UnifiedState and LogicEngine.

    Provides logic operations and manages engine instances per dimension.
    """

    def __init__(self):
        """Initialize bridge with engine cache"""
        self._engines = {}

    def get_engine(self, dimension: int):
        """Get or create LogicEngine for dimension."""
        if dimension not in self._engines:
            from logic.heyting import LogicEngine
            self._engines[dimension] = LogicEngine(dimension)
        return self._engines[dimension]

    def meet(self, state_a, state_b):
        """Compute meet (∧) operation."""
        if state_a.dimension != state_b.dimension:
            raise ValueError("Dimension mismatch")
        engine = self.get_engine(state_a.dimension)
        return engine.meet(state_a, state_b)

    def join(self, state_a, state_b):
        """Compute join (∨) operation."""
        if state_a.dimension != state_b.dimension:
            raise ValueError("Dimension mismatch")
        engine = self.get_engine(state_a.dimension)
        return engine.join(state_a, state_b)

    def negate(self, state):
        """Compute negation (¬) operation."""
        engine = self.get_engine(state.dimension)
        return engine.negate(state)

    def implies(self, state_a, state_b):
        """Compute implication (→) operation."""
        if state_a.dimension != state_b.dimension:
            raise ValueError("Dimension mismatch")
        engine = self.get_engine(state_a.dimension)
        return engine.implies(state_a, state_b)


# Global logic bridge instance
_logic_bridge = None

def get_logic_bridge():
    """Get global LogicBridge instance"""
    global _logic_bridge
    if _logic_bridge is None:
        _logic_bridge = LogicBridge()
    return _logic_bridge


# ============================================================================
# GRAPH BRIDGE
# ============================================================================

class GraphBridge:
    """
    Bridge between UnifiedState and GraphEngine.

    Provides conversion methods and manages engine instances per dimension.
    """

    def __init__(self):
        """Initialize bridge with engine cache"""
        self._engines = {}

    def get_engine(self, dimension: int):
        """Get or create GraphEngine for dimension."""
        if dimension not in self._engines:
            from graph.engine import GraphEngine
            self._engines[dimension] = GraphEngine(dimension)
        return self._engines[dimension]

    def state_to_graph(self, state, include_zero_features: bool = True,
                      edge_threshold: float = 1e-10):
        """Convert UnifiedState to PyG graph."""
        engine = self.get_engine(state.dimension)
        return engine.state_to_graph(state, include_zero_features, edge_threshold)

    def graph_to_state(self, graph, dimension: int):
        """Convert PyG graph to UnifiedState."""
        engine = self.get_engine(dimension)
        return engine.graph_to_state(graph)

    def get_node_info(self, graph, node_idx: int, dimension: int):
        """Get information about a specific node."""
        engine = self.get_engine(dimension)
        return engine.get_node_info(graph, node_idx)

    def get_neighbors(self, graph, node_idx: int, dimension: int):
        """Get neighbor nodes."""
        engine = self.get_engine(dimension)
        return engine.get_neighbors(graph, node_idx)


# Global graph bridge instance
_graph_bridge = None

def get_graph_bridge():
    """Get global GraphBridge instance"""
    global _graph_bridge
    if _graph_bridge is None:
        _graph_bridge = GraphBridge()
    return _graph_bridge
