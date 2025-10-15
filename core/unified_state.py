"""
unified_state.py

Core state management and verification

Single source of truth for unified representations.
Maintains canonical tensor representation with lazy view computation.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from enum import Enum


class LogicType(Enum):
    """Logic system type based on dimension"""
    HEYTING = "heyting"      # 1D: Intuitionistic logic
    BOOLEAN = "boolean"       # 2D+: Classical logic


class UnifiedState:
    """
    Single source of truth for unified representations.

    Maintains a canonical tensor representation and lazily computes
    views as Clifford elements, graph structures, or logic elements.

    Key Properties:
    - Dimension-aware (1D → Heyting, 2D+ → Boolean)
    - Lazy view computation with caching
    - Automatic consistency verification
    - GPU-accelerated when available

    Attributes:
        primary_data: Canonical tensor representation (blade coefficients)
        dimension: Spatial dimension (1, 2, or 3)
        logic_type: Heyting (1D) or Boolean (2D+)
        cached_views: Lazy-computed representation views
        device: Computation device (CPU/CUDA)
    """

    def __init__(
        self,
        data: torch.Tensor,
        dimension: int,
        device: Optional[torch.device] = None,
        verify: bool = True
    ):
        """Initialize unified state."""
        if dimension not in [1, 2, 3]:
            raise ValueError(f"Dimension must be 1, 2, or 3. Got: {dimension}")

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.primary_data = data.to(self.device)
        self.dimension = dimension
        self.logic_type = LogicType.HEYTING if dimension == 1 else LogicType.BOOLEAN
        self.cached_views: Dict[str, Any] = {}
        self.expected_size = 2 ** dimension

        if verify:
            self._verify_data_shape()

    def _verify_data_shape(self) -> None:
        """Verify primary data has correct shape for dimension"""
        if self.primary_data.numel() != self.expected_size:
            raise ValueError(
                f"Data size {self.primary_data.numel()} incompatible with "
                f"dimension {self.dimension} (expected {self.expected_size})"
            )

    @classmethod
    def from_vector(cls, coefficients: list, dimension: Optional[int] = None,
                    device: Optional[torch.device] = None) -> 'UnifiedState':
        """Create state from vector coefficients."""
        data = torch.tensor(coefficients, dtype=torch.float32)

        if dimension is None:
            size = data.numel()
            if size == 2:
                dimension = 1
            elif size == 4:
                dimension = 2
            elif size == 8:
                dimension = 3
            else:
                raise ValueError(f"Cannot infer dimension from size {size}")

        return cls(data, dimension, device)

    @classmethod
    def zero(cls, dimension: int, device: Optional[torch.device] = None) -> 'UnifiedState':
        """Create zero state (additive identity)"""
        size = 2 ** dimension
        data = torch.zeros(size, dtype=torch.float32)
        return cls(data, dimension, device, verify=False)

    @classmethod
    def scalar(cls, value: float, dimension: int,
               device: Optional[torch.device] = None) -> 'UnifiedState':
        """Create scalar state (grade 0)"""
        size = 2 ** dimension
        data = torch.zeros(size, dtype=torch.float32)
        data[0] = value
        return cls(data, dimension, device, verify=False)

    def invalidate_cache(self, view_name: Optional[str] = None) -> None:
        """Invalidate cached views."""
        if view_name is None:
            self.cached_views.clear()
        elif view_name in self.cached_views:
            del self.cached_views[view_name]

    def as_clifford(self):
        """Get Clifford algebra representation (deferred to CliffordEngine)."""
        if 'clifford' in self.cached_views:
            return self.cached_views['clifford']
        raise NotImplementedError("Clifford conversion requires CliffordEngine")

    def as_graph(self):
        """Get graph representation (deferred to GraphEngine)."""
        if 'graph' in self.cached_views:
            return self.cached_views['graph']
        raise NotImplementedError("Graph conversion requires GraphEngine")

    def as_logic(self):
        """Get logic representation (deferred to LogicEngine)."""
        if 'logic' in self.cached_views:
            return self.cached_views['logic']
        raise NotImplementedError("Logic conversion requires LogicEngine")

    def get_coefficients(self) -> torch.Tensor:
        """Get raw blade coefficients"""
        return self.primary_data.clone()

    def get_grade(self, grade: int) -> torch.Tensor:
        """Extract coefficients for specific grade."""
        if grade < 0 or grade > self.dimension:
            raise ValueError(f"Grade {grade} invalid for dimension {self.dimension}")
        return self.primary_data[grade:grade+1]

    def norm(self) -> float:
        """Compute Euclidean norm of state"""
        return torch.norm(self.primary_data).item()

    def is_zero(self, epsilon: float = 1e-10) -> bool:
        """Check if state is approximately zero"""
        return self.norm() < epsilon

    def __repr__(self) -> str:
        """String representation"""
        logic_str = self.logic_type.value.capitalize()
        return (f"UnifiedState(dim={self.dimension}, logic={logic_str}, "
                f"norm={self.norm():.4f}, device={self.device})")

    def __eq__(self, other: 'UnifiedState') -> bool:
        """Equality check"""
        if not isinstance(other, UnifiedState):
            return False
        return (self.dimension == other.dimension and
                torch.allclose(self.primary_data, other.primary_data))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            'data': self.primary_data.cpu().tolist(),
            'dimension': self.dimension,
            'logic_type': self.logic_type.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedState':
        """Deserialize from dictionary"""
        tensor_data = torch.tensor(data['data'], dtype=torch.float32)
        return cls(tensor_data, data['dimension'])
