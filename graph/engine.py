"""
engine.py

Graph neural network components

Converts Clifford algebra elements to graph structures for GNN processing.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data

from core.unified_state import UnifiedState
from ga_clifford.engine import CliffordEngine


class GraphEngine:
    """
    Graph representation engine for Clifford algebra elements.

    Converts multivectors to graph structures:
    - Nodes: Basis blades
    - Node features: Blade coefficients + grade information
    - Edges: Geometric product relationships
    - Edge features: Product coefficients
    """

    def __init__(self, dimension: int):
        """Initialize graph engine for given dimension."""
        if dimension not in [1, 2, 3]:
            raise ValueError(f"Dimension must be 1, 2, or 3. Got: {dimension}")

        self.dimension = dimension
        self.num_blades = 2 ** dimension
        self.clifford_engine = CliffordEngine(dimension)
        self.blade_names = self.clifford_engine.blade_names
        self.blade_grades = self._compute_blade_grades()
        self.product_structure = self._compute_product_structure()

    def _compute_blade_grades(self) -> List[int]:
        """Compute grade for each blade."""
        grades = []
        for name in self.blade_names:
            if name == '':
                grade = 0
            else:
                grade = len([c for c in name if c.isdigit()])
            grades.append(grade)
        return grades

    def _compute_product_structure(self) -> Dict[Tuple[int, int], Tuple[int, float]]:
        """Precompute geometric product structure."""
        structure = {}

        for i in range(self.num_blades):
            for j in range(self.num_blades):
                blade_i = torch.zeros(self.num_blades)
                blade_i[i] = 1.0

                blade_j = torch.zeros(self.num_blades)
                blade_j[j] = 1.0

                result = self.clifford_engine.geometric_product(blade_i, blade_j)

                max_idx = torch.argmax(torch.abs(result)).item()
                max_coeff = result[max_idx].item()

                if abs(max_coeff) > 1e-10:
                    structure[(i, j)] = (max_idx, max_coeff)

        return structure

    def state_to_graph(self, state: UnifiedState, include_zero_features: bool = True,
                      edge_threshold: float = 1e-10) -> Data:
        """Convert UnifiedState to PyG graph."""
        if state.dimension != self.dimension:
            raise ValueError(f"Dimension mismatch")

        coeffs = state.primary_data.detach().cpu()

        # Node features: [coefficient, grade, is_scalar]
        node_features = []
        for i in range(self.num_blades):
            coeff = coeffs[i].item()
            grade = self.blade_grades[i]
            is_scalar = 1.0 if grade == 0 else 0.0
            node_features.append([coeff, grade, is_scalar])

        x = torch.tensor(node_features, dtype=torch.float32)

        # Build edges
        edge_index = []
        edge_attr = []

        for (i, j), (k, coeff) in self.product_structure.items():
            if abs(coeff) > edge_threshold:
                edge_index.append([i, j])
                edge_attr.append([coeff])

        if len(edge_index) == 0:
            edge_index = [[i, i] for i in range(self.num_blades)]
            edge_attr = [[1.0] for _ in range(self.num_blades)]

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                   num_nodes=self.num_blades)
        data.dimension = self.dimension
        data.blade_names = self.blade_names
        data.original_coeffs = coeffs

        return data

    def graph_to_state(self, data: Data) -> UnifiedState:
        """Convert PyG graph back to UnifiedState."""
        coeffs = data.x[:, 0]
        return UnifiedState(coeffs, self.dimension)

    def get_node_info(self, data: Data, node_idx: int) -> Dict:
        """Get information about a specific node."""
        if node_idx >= data.num_nodes:
            raise ValueError(f"Node {node_idx} out of range")

        features = data.x[node_idx]
        return {
            'index': node_idx,
            'blade_name': self.blade_names[node_idx],
            'coefficient': features[0].item(),
            'grade': int(features[1].item()),
            'is_scalar': bool(features[2].item() > 0.5),
        }

    def get_neighbors(self, data: Data, node_idx: int) -> List[int]:
        """Get neighbor nodes for a given node."""
        mask = data.edge_index[0] == node_idx
        neighbors = data.edge_index[1][mask].tolist()
        return list(set(neighbors))
