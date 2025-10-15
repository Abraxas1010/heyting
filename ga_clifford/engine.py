"""
engine.py

Clifford algebra operations

Handles geometric algebra operations including wedge, inner,
and geometric products for 1D, 2D, and 3D algebras.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any
import os

# Ensure numba is disabled
os.environ['NUMBA_DISABLE_JIT'] = '1'

from clifford import Cl


class CliffordEngine:
    """
    Clifford algebra operations engine.

    Handles geometric algebra operations including:
    - Wedge product (exterior product: ∧)
    - Inner product (contraction: ·)
    - Geometric product (full product)
    - Grade projection and extraction
    - Rotor creation and application

    Supports 1D, 2D, and 3D algebras with proper grade handling.
    """

    def __init__(self, dimension: int):
        """Initialize Clifford engine for given dimension."""
        if dimension not in [1, 2, 3]:
            raise ValueError(f"Dimension must be 1, 2, or 3. Got: {dimension}")

        self.dimension = dimension
        self.layout, self.blades = Cl(dimension)
        self.blade_names = self._get_blade_names()
        self._basis_cache: Dict[str, Any] = {}

    def _get_blade_names(self) -> list:
        """Get ordered list of blade names for this dimension."""
        if self.dimension == 1:
            return ['', 'e1']
        elif self.dimension == 2:
            return ['', 'e1', 'e2', 'e12']
        else:  # dimension == 3
            return ['', 'e1', 'e2', 'e3', 'e12', 'e13', 'e23', 'e123']

    def tensor_to_multivector(self, tensor: torch.Tensor):
        """Convert tensor of coefficients to Clifford multivector."""
        expected_size = 2 ** self.dimension
        if tensor.numel() != expected_size:
            raise ValueError(
                f"Tensor size {tensor.numel()} incompatible with "
                f"dimension {self.dimension} (expected {expected_size})"
            )

        # Convert tensor to numpy for clifford
        coeffs = tensor.detach().cpu().numpy().flatten()

        # Build multivector directly (more reliable than loop with +=)
        # Sum all blade components in one expression
        mv_terms = [coeffs[i] * self.blades[name]
                    for i, name in enumerate(self.blade_names)
                    if abs(coeffs[i]) > 1e-15]

        if not mv_terms:
            # All coefficients zero
            return self.layout.scalar(0)

        # Sum all terms at once
        mv = sum(mv_terms[1:], mv_terms[0]) if len(mv_terms) > 1 else mv_terms[0]

        return mv

    def multivector_to_tensor(self, mv) -> torch.Tensor:
        """Convert Clifford multivector to tensor of coefficients."""
        # Extract coefficient array from multivector
        # The .value attribute gives numpy array in canonical order
        if hasattr(mv, 'value'):
            coeffs = mv.value
        else:
            # Fallback for older clifford versions
            coeffs = np.array([float(mv[(i,)]) for i in range(len(self.blade_names))])

        return torch.tensor(coeffs, dtype=torch.float32)

    def wedge_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute wedge (exterior) product: a ∧ b."""
        mv_a = self.tensor_to_multivector(a)
        mv_b = self.tensor_to_multivector(b)
        result_mv = mv_a ^ mv_b
        return self.multivector_to_tensor(result_mv)

    def inner_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute inner (dot) product: a · b."""
        mv_a = self.tensor_to_multivector(a)
        mv_b = self.tensor_to_multivector(b)
        result_mv = mv_a | mv_b
        return self.multivector_to_tensor(result_mv)

    def geometric_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute geometric product: ab."""
        mv_a = self.tensor_to_multivector(a)
        mv_b = self.tensor_to_multivector(b)
        result_mv = mv_a * mv_b
        return self.multivector_to_tensor(result_mv)

    def grade_project(self, tensor: torch.Tensor, grade: int) -> torch.Tensor:
        """Project multivector onto specific grade."""
        if grade < 0 or grade > self.dimension:
            raise ValueError(f"Grade {grade} invalid for dimension {self.dimension}")

        mv = self.tensor_to_multivector(tensor)
        result_mv = mv(grade)
        return self.multivector_to_tensor(result_mv)

    def reverse(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute reverse (dagger) of multivector."""
        mv = self.tensor_to_multivector(tensor)
        result_mv = ~mv
        return self.multivector_to_tensor(result_mv)

    def dual(self, tensor: torch.Tensor) -> torch.Tensor:
        """Compute Hodge dual of multivector."""
        mv = self.tensor_to_multivector(tensor)

        if self.dimension == 1:
            I = self.blades['e1']
        elif self.dimension == 2:
            I = self.blades['e12']
        else:
            I = self.blades['e123']

        result_mv = mv * I
        return self.multivector_to_tensor(result_mv)

    def norm(self, tensor: torch.Tensor) -> float:
        """Compute norm of multivector."""
        return torch.norm(tensor).item()

    def create_rotor(self, angle: float, plane_bivector: torch.Tensor) -> torch.Tensor:
        """Create rotor for rotation in specified plane."""
        plane_norm = self.norm(plane_bivector)
        if plane_norm < 1e-10:
            raise ValueError("Plane bivector must be non-zero")

        normalized_plane = plane_bivector / plane_norm
        half_angle = angle / 2

        rotor = torch.zeros_like(plane_bivector)
        rotor[0] = np.cos(half_angle)
        rotor -= np.sin(half_angle) * normalized_plane

        return rotor

    def apply_rotor(self, rotor: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """Apply rotor to rotate a vector: v' = R v R†"""
        rv = self.geometric_product(rotor, vector)
        rotor_reverse = self.reverse(rotor)
        result = self.geometric_product(rv, rotor_reverse)
        return result
