"""
combinatorial_engine.py

Master engine for combinatorial reasoning across representations and modes.
Coordinates meta-level path optimization through 12 computational modalities.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from itertools import product

from training.reasoning_engines import (
    Representation, ReasoningMode,
    InductionEngine, DeductionEngine, AbductionEngine
)
from core.unified_state import UnifiedState
from bridges.logic_clifford import get_clifford_bridge, get_logic_bridge, get_graph_bridge


@dataclass
class CombinatorialNode:
    """Single computational modality: (Representation, ReasoningMode)"""
    representation: Representation
    mode: ReasoningMode

    def __repr__(self):
        return f"({self.representation.value},{self.mode.value})"

    def __hash__(self):
        return hash((self.representation, self.mode))

    def __eq__(self, other):
        return (self.representation == other.representation and
                self.mode == other.mode)


@dataclass
class CombinatorialPath:
    """Sequence of computational modalities"""
    nodes: List[CombinatorialNode]

    def __repr__(self):
        return "→".join(str(node) for node in self.nodes)

    def __len__(self):
        return len(self.nodes)


class CombinatorialReasoningEngine:
    """
    Master coordinator for tripartite cognitive architecture.

    12 computational modalities: (U,C,L,G) × (I,D,A)
    144 possible single-hop transitions
    """

    def __init__(self, dimension: int = 2):
        self.dimension = dimension

        # Layer 0: Topos
        self.topos = UnifiedState

        # Layer 1: Sheaves
        self.sheaves = {
            Representation.CLIFFORD: get_clifford_bridge(),
            Representation.LOGIC: get_logic_bridge(),
            Representation.GRAPH: get_graph_bridge(),
        }

        # Layer 2: Reasoning engines
        self.reasoning_engines = {
            ReasoningMode.INDUCTION: InductionEngine(),
            ReasoningMode.DEDUCTION: DeductionEngine(),
            ReasoningMode.ABDUCTION: AbductionEngine(),
        }

        # Meta-level tracking
        self.path_registry = {}
        self.path_performance = {}

    def execute_node(self, node: CombinatorialNode, state: UnifiedState,
                     context: Dict[str, Any]) -> Tuple[UnifiedState, Dict[str, Any]]:
        """
        Execute single (Representation, Mode) node.

        Returns: (new_state, updated_context)
        """
        # Step 1: Convert to target representation
        if node.representation == Representation.CLIFFORD:
            view = state.as_clifford()
        elif node.representation == Representation.LOGIC:
            view = state.as_logic()
        elif node.representation == Representation.GRAPH:
            view = state.as_graph()
        else:  # UNIFIED
            view = state

        # Step 2: Apply reasoning mode
        engine = self.reasoning_engines[node.mode]
        result_view, new_context = engine.apply(view, node.representation, context)

        # Step 3: Convert back to UnifiedState
        if node.representation == Representation.CLIFFORD:
            result_state = self.sheaves[Representation.CLIFFORD].clifford_to_state(
                result_view, self.dimension
            )
        elif node.representation == Representation.LOGIC:
            result_state = result_view
        elif node.representation == Representation.GRAPH:
            result_state = self.sheaves[Representation.GRAPH].graph_to_state(
                result_view, self.dimension
            )
        else:
            result_state = result_view

        return result_state, new_context

    def execute_path(self, path: CombinatorialPath, initial_state: UnifiedState,
                     context: Dict[str, Any]) -> Tuple[UnifiedState, Dict[str, Any]]:
        """Execute full path through combinatorial space"""
        current_state = initial_state
        current_context = context.copy()
        path_log = []

        for i, node in enumerate(path.nodes):
            current_state, current_context = self.execute_node(
                node, current_state, current_context
            )
            path_log.append({'node': node, 'state': current_state})

        self.path_registry[str(path)] = path_log
        return current_state, current_context

    def generate_diverse_paths(self, max_length: int = 4) -> List[CombinatorialPath]:
        """Generate diverse exploration paths"""
        paths = []

        # Strategy 1: Mode progression within representation
        for rep in Representation:
            nodes = [
                CombinatorialNode(rep, ReasoningMode.INDUCTION),
                CombinatorialNode(rep, ReasoningMode.DEDUCTION),
                CombinatorialNode(rep, ReasoningMode.ABDUCTION),
            ]
            paths.append(CombinatorialPath(nodes))

        # Strategy 2: Representation cycle with fixed mode
        for mode in ReasoningMode:
            nodes = [
                CombinatorialNode(Representation.UNIFIED, mode),
                CombinatorialNode(Representation.CLIFFORD, mode),
                CombinatorialNode(Representation.LOGIC, mode),
                CombinatorialNode(Representation.GRAPH, mode),
            ]
            paths.append(CombinatorialPath(nodes))

        # Strategy 3: Hybrid learning cycle
        paths.append(CombinatorialPath([
            CombinatorialNode(Representation.GRAPH, ReasoningMode.INDUCTION),
            CombinatorialNode(Representation.CLIFFORD, ReasoningMode.DEDUCTION),
            CombinatorialNode(Representation.LOGIC, ReasoningMode.ABDUCTION),
            CombinatorialNode(Representation.GRAPH, ReasoningMode.INDUCTION),
        ]))

        # Strategy 4: Geometric-first
        paths.append(CombinatorialPath([
            CombinatorialNode(Representation.CLIFFORD, ReasoningMode.DEDUCTION),
            CombinatorialNode(Representation.LOGIC, ReasoningMode.DEDUCTION),
            CombinatorialNode(Representation.GRAPH, ReasoningMode.INDUCTION),
        ]))

        # Strategy 5: Abduction-driven
        paths.append(CombinatorialPath([
            CombinatorialNode(Representation.LOGIC, ReasoningMode.ABDUCTION),
            CombinatorialNode(Representation.CLIFFORD, ReasoningMode.DEDUCTION),
            CombinatorialNode(Representation.GRAPH, ReasoningMode.INDUCTION),
            CombinatorialNode(Representation.LOGIC, ReasoningMode.ABDUCTION),
        ]))

        return paths

    def explore_combinatorial_space(self, n_samples: int = 10,
                                   max_path_length: int = 4):
        """Systematically explore and benchmark paths"""
        print("="*60)
        print("EXPLORING COMBINATORIAL REASONING SPACE")
        print("="*60)

        paths = self.generate_diverse_paths(max_path_length)
        print(f"\nGenerated {len(paths)} diverse paths")
        print(f"Testing on {n_samples} samples")

        for path_idx, path in enumerate(paths):
            print(f"\n{'='*60}")
            print(f"PATH {path_idx + 1}/{len(paths)}: {path}")
            print(f"{'='*60}")

            path_errors = []

            for sample_idx in range(n_samples):
                state = self._generate_test_state()
                target = self._generate_target(state)

                context = {
                    'target': target,
                    'dimension': self.dimension,
                    'training_samples': self._generate_training_samples(10)
                }

                try:
                    result_state, _ = self.execute_path(path, state, context)
                    error = torch.norm(
                        result_state.primary_data - target.primary_data
                    ).item()
                    path_errors.append(error)
                except Exception as e:
                    print(f"    ⚠ Sample {sample_idx + 1} failed: {e}")
                    path_errors.append(float('inf'))

            avg_error = sum(path_errors) / len(path_errors)
            self.path_performance[str(path)] = {
                'errors': path_errors,
                'mean': avg_error,
                'std': torch.std(torch.tensor(path_errors)).item()
            }

            print(f"  Performance: {avg_error:.6f}")

        self._analyze_performance()

    def _analyze_performance(self):
        """Analyze which paths work best"""
        print("\n" + "="*60)
        print("PATH PERFORMANCE ANALYSIS")
        print("="*60)

        sorted_paths = sorted(self.path_performance.items(),
                             key=lambda x: x[1]['mean'])

        print("\nTop 5 Best Paths:")
        for i, (path, perf) in enumerate(sorted_paths[:5]):
            print(f"\n{i+1}. {path}")
            print(f"   Error: {perf['mean']:.6f} ± {perf['std']:.6f}")

    def _generate_test_state(self) -> UnifiedState:
        coeffs = torch.randn(2 ** self.dimension)
        return UnifiedState(coeffs, self.dimension)

    def _generate_target(self, state: UnifiedState) -> UnifiedState:
        other = self._generate_test_state()
        bridge = get_clifford_bridge()
        return bridge.wedge_product(state, other)

    def _generate_training_samples(self, n: int) -> List[dict]:
        samples = []
        bridge = get_clifford_bridge()
        for _ in range(n):
            a = self._generate_test_state()
            b = self._generate_test_state()
            target = bridge.wedge_product(a, b)
            samples.append({'data': a, 'answer': target})
        return samples
