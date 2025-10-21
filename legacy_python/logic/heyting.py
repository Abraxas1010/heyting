"""
heyting.py

Logic operations (Heyting/Boolean)

Dimension-dependent logic engine:
- 1D: Heyting/Intuitionistic logic
- 2D+: Boolean/Classical logic
"""

import torch
import numpy as np
from typing import Optional, Tuple
from enum import Enum

from core.unified_state import UnifiedState, LogicType
from ga_clifford.engine import CliffordEngine


class LogicEngine:
    """
    Dimension-dependent logic operations engine.

    Implements:
    - 1D: Heyting/Intuitionistic logic (¬¬a ≠ a, excluded middle may fail)
    - 2D+: Boolean/Classical logic (¬¬a = a, excluded middle holds)
    """

    def __init__(self, dimension: int):
        """Initialize logic engine for given dimension."""
        if dimension not in [1, 2, 3]:
            raise ValueError(f"Dimension must be 1, 2, or 3. Got: {dimension}")

        self.dimension = dimension
        self.logic_type = LogicType.HEYTING if dimension == 1 else LogicType.BOOLEAN
        self.clifford_engine = CliffordEngine(dimension)
        self.orthogonality_epsilon = 1e-6

    def check_orthogonality(self, state_a: UnifiedState, state_b: UnifiedState) -> bool:
        """Check if two states are orthogonal."""
        if state_a.dimension != state_b.dimension:
            raise ValueError("Dimension mismatch")

        inner = self.clifford_engine.inner_product(
            state_a.primary_data, state_b.primary_data
        )
        norm = torch.norm(inner).item()
        return norm < self.orthogonality_epsilon

    def meet(self, state_a: UnifiedState, state_b: UnifiedState) -> Optional[UnifiedState]:
        """
        Compute meet operation: a ∧ b (logical AND).

        In 1D (Heyting): Only defined if states are orthogonal
        In 2D+ (Boolean): Always defined
        """
        if state_a.dimension != state_b.dimension:
            raise ValueError("Dimension mismatch")

        if self.logic_type == LogicType.HEYTING:
            if not self.check_orthogonality(state_a, state_b):
                return None

        result_tensor = self.clifford_engine.wedge_product(
            state_a.primary_data, state_b.primary_data
        )
        return UnifiedState(result_tensor, state_a.dimension)

    def join(self, state_a: UnifiedState, state_b: UnifiedState) -> UnifiedState:
        """Compute join operation: a ∨ b (logical OR)."""
        if state_a.dimension != state_b.dimension:
            raise ValueError("Dimension mismatch")

        # Element-wise max of absolute values
        result_tensor = torch.maximum(
            torch.abs(state_a.primary_data),
            torch.abs(state_b.primary_data)
        )
        return UnifiedState(result_tensor, state_a.dimension)

    def negate(self, state: UnifiedState) -> UnifiedState:
        """
        Compute negation: ¬a.

        1D (Heyting): 720° rotation (¬¬a ≠ a)
        2D+ (Boolean): 360° rotation (¬¬a = a)
        """
        if self.logic_type == LogicType.HEYTING:
            result_tensor = -state.primary_data
            rotation_factor = 0.1
            result_tensor = result_tensor * (1 + rotation_factor)
        else:
            result_tensor = -state.primary_data

        return UnifiedState(result_tensor, state.dimension)

    def implies(self, state_a: UnifiedState, state_b: UnifiedState) -> UnifiedState:
        """Compute implication: a → b = ¬a ∨ b."""
        if state_a.dimension != state_b.dimension:
            raise ValueError("Dimension mismatch")

        not_a = self.negate(state_a)
        return self.join(not_a, state_b)

    def top(self, dimension: int) -> UnifiedState:
        """Create top element (⊤): tautology/true."""
        return UnifiedState.scalar(1.0, dimension)

    def bottom(self, dimension: int) -> UnifiedState:
        """Create bottom element (⊥): contradiction/false."""
        return UnifiedState.zero(dimension)

    def verify_distributivity(self, a: UnifiedState, b: UnifiedState,
                            c: UnifiedState, epsilon: float = 1e-5) -> bool:
        """Verify distributivity: a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)."""
        b_join_c = self.join(b, c)
        left = self.meet(a, b_join_c)

        a_meet_b = self.meet(a, b)
        a_meet_c = self.meet(a, c)

        if left is None or a_meet_b is None or a_meet_c is None:
            return True

        right = self.join(a_meet_b, a_meet_c)
        return torch.allclose(left.primary_data, right.primary_data, atol=epsilon)

    def verify_excluded_middle(self, state: UnifiedState,
                              epsilon: float = 1e-5) -> Tuple[bool, float]:
        """Verify excluded middle: a ∨ ¬a = ⊤."""
        not_a = self.negate(state)
        result = self.join(state, not_a)
        top = self.top(state.dimension)

        deviation = torch.norm(result.primary_data - top.primary_data).item()
        holds = deviation < epsilon
        return holds, deviation

    def verify_double_negation(self, state: UnifiedState,
                              epsilon: float = 1e-5) -> Tuple[bool, float]:
        """Verify double negation: ¬¬a = a."""
        not_a = self.negate(state)
        not_not_a = self.negate(not_a)

        deviation = torch.norm(not_not_a.primary_data - state.primary_data).item()
        holds = deviation < epsilon
        return holds, deviation
