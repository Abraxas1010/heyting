"""
graph/layers.py - Geometric Message Passing Layer
Task 18: Message passing preserving Clifford structure and respecting Logic constraints
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Tuple
import sys
from pathlib import Path

project_root = Path("/content/drive/MyDrive/generative_ontology")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import generative_ontology as go


class GeometricMessagePassing(MessagePassing):
    """
    Message passing layer respecting all three algebraic structures.
    
    Integration:
    - Clifford: Messages preserve grade structure via geometric product
    - Logic: Check operation validity before passing (Heyting constraints)
    - Graph: Standard PyG message passing framework
    """
    
    def __init__(
        self,
        dimension: int,
        aggr: str = 'add',
        flow: str = 'source_to_target',
        orthogonality_epsilon: float = 1e-6,
        **kwargs
    ):
        super().__init__(aggr=aggr, flow=flow, **kwargs)
        
        if dimension not in [1, 2, 3]:
            raise ValueError(f"Dimension must be 1, 2, or 3, got {dimension}")
        
        self.dimension = dimension
        self.orthogonality_epsilon = orthogonality_epsilon
        self.expected_size = 2 ** dimension
        
        # Initialize all three engines
        self.clifford_engine = go.CliffordEngine(dimension)
        self.logic_engine = go.LogicEngine(dimension)
        self.graph_bridge = go.get_graph_bridge()
        self.clifford_bridge = go.get_clifford_bridge()
        
        self.is_heyting = (dimension == 1)
        self.device = torch.device('cpu')
        
        # Learnable weights
        self.message_transform = nn.Linear(self.expected_size, self.expected_size)
        self.message_transform.to(self.device)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through geometric message passing."""
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.device)
        
        if x.size(1) != self.expected_size:
            raise ValueError(f"Expected node features of size {self.expected_size}, got {x.size(1)}")
        
        edge_index, edge_attr = self._add_self_loops(edge_index, edge_attr, x.size(0))
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j: torch.Tensor, x_i: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate messages using geometric product with Logic constraints."""
        batch_size = x_j.size(0)
        device = x_j.device
        messages = torch.zeros(batch_size, self.expected_size, device=device)
        
        for i in range(batch_size):
            state_j = go.UnifiedState(x_j[i].cpu(), self.dimension)
            state_i = go.UnifiedState(x_i[i].cpu(), self.dimension)
            
            # === LOGIC ENGINE: Check operation validity ===
            if self.is_heyting:
                is_orthogonal = self.logic_engine.check_orthogonality(state_j, state_i)
                if not is_orthogonal:
                    continue
            
            # === CLIFFORD ENGINE: Compute geometric product ===
            try:
                if edge_attr is not None:
                    edge_state = go.UnifiedState(edge_attr[i].cpu(), self.dimension)
                    intermediate = self.clifford_bridge.geometric_product(state_j, edge_state)
                    result = self.clifford_bridge.geometric_product(intermediate, state_i)
                else:
                    result = self.clifford_bridge.geometric_product(state_j, state_i)
                
                result_data = result.primary_data.to(device)
                transformed = self.message_transform(result_data.unsqueeze(0))
                messages[i] = transformed.squeeze(0)
                
            except Exception as e:
                print(f"Warning: Message computation failed at edge {i}: {e}")
                continue
        
        return messages
    
    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor, 
                  ptr: Optional[torch.Tensor] = None, 
                  dim_size: Optional[int] = None) -> torch.Tensor:
        """Aggregate messages respecting grade structure."""
        aggregated = super().aggregate(inputs, index, ptr, dim_size)
        
        # === CLIFFORD ENGINE: Separate by grade ===
        num_nodes = aggregated.size(0)
        device = aggregated.device
        grade_separated = torch.zeros_like(aggregated)
        
        for node_idx in range(num_nodes):
            state = go.UnifiedState(aggregated[node_idx].cpu(), self.dimension)
            
            grade_components = []
            for grade in range(self.dimension + 1):
                grade_proj = self.clifford_engine.grade_project(
                    state.primary_data, grade=grade
                )
                grade_components.append(grade_proj)
            
            # === LOGIC ENGINE: Combine grades using join ===
            if len(grade_components) > 1:
                combined = grade_components[0]
                for comp in grade_components[1:]:
                    comp_state = go.UnifiedState(comp.cpu(), self.dimension)
                    combined_state = go.UnifiedState(combined.cpu(), self.dimension)
                    joined = self.logic_engine.join(combined_state, comp_state)
                    combined = joined.primary_data.to(device)
                
                grade_separated[node_idx] = combined
            else:
                grade_separated[node_idx] = grade_components[0]
        
        return grade_separated
    
    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Update nodes using logical implication."""
        num_nodes = x.size(0)
        device = x.device
        updated = torch.zeros_like(x)
        
        for node_idx in range(num_nodes):
            current_state = go.UnifiedState(x[node_idx].cpu(), self.dimension)
            aggregated_state = go.UnifiedState(aggr_out[node_idx].cpu(), self.dimension)
            
            # === LOGIC ENGINE: Compute implication ===
            try:
                implied = self.logic_engine.implies(current_state, aggregated_state)
                updated[node_idx] = implied.primary_data.to(device)
            except Exception as e:
                print(f"Warning: Implication failed at node {node_idx}: {e}")
                updated[node_idx] = aggr_out[node_idx]
        
        return updated
    
    def _add_self_loops(self, edge_index: torch.Tensor, 
                        edge_attr: Optional[torch.Tensor], 
                        num_nodes: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Add self-loops to edge index."""
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, fill_value='mean', num_nodes=num_nodes
        )
        return edge_index, edge_attr


class LogicAwareConv(nn.Module):
    """
    Convolution layer with dimension-dependent logic behavior.
    
    - 1D (Heyting): Checks orthogonality before operations
    - 2D+ (Boolean): All operations valid
    - All dimensions: Equivariant to Clifford rotations
    """
    
    def __init__(self, dimension: int, in_channels: Optional[int] = None, 
                 out_channels: Optional[int] = None, use_wedge: bool = False):
        super().__init__()
        
        if dimension not in [1, 2, 3]:
            raise ValueError(f"Dimension must be 1, 2, or 3, got {dimension}")
        
        self.dimension = dimension
        self.expected_size = 2 ** dimension
        self.in_channels = in_channels or self.expected_size
        self.out_channels = out_channels or self.expected_size
        
        if self.in_channels != self.expected_size:
            raise ValueError(f"in_channels must equal 2^dimension={self.expected_size}")
        if self.out_channels != self.expected_size:
            raise ValueError(f"out_channels must equal 2^dimension={self.expected_size}")
        
        self.use_wedge = use_wedge
        
        # Initialize engines
        self.logic_engine = go.LogicEngine(dimension)
        self.clifford_engine = go.CliffordEngine(dimension)
        self.clifford_bridge = go.get_clifford_bridge()
        
        self.is_heyting = (dimension == 1)
        self.device = torch.device('cpu')
        
        # Grade-separated learnable weights
        self.grade_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(self.dimension + 1)
        ])
        self.bias = nn.Parameter(torch.zeros(self.out_channels))
        
        self._build_grade_mapping()
        self.to(self.device)
    
    def _build_grade_mapping(self):
        """Build mapping from grades to blade indices."""
        self.grade_to_blades = {}
        for grade in range(self.dimension + 1):
            blade_indices = []
            for idx, name in enumerate(self.clifford_engine.blade_names):
                if name == '':
                    blade_grade = 0
                else:
                    blade_grade = len(name) - name.count('e')
                
                if blade_grade == grade:
                    blade_indices.append(idx)
            
            self.grade_to_blades[grade] = blade_indices
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                apply_constraints: bool = True) -> torch.Tensor:
        """Forward pass with logic-dependent behavior."""
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        
        num_nodes = x.size(0)
        
        if self.is_heyting and apply_constraints:
            output = self._heyting_forward(x, edge_index)
        else:
            output = self._boolean_forward(x, edge_index)
        
        return output
    
    def _heyting_forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Heyting-constrained forward pass (1D only)."""
        num_nodes = x.size(0)
        output = torch.zeros_like(x)
        
        for node_idx in range(num_nodes):
            neighbor_mask = edge_index[1] == node_idx
            neighbor_indices = edge_index[0, neighbor_mask]
            
            if len(neighbor_indices) == 0:
                output[node_idx] = self._apply_transformation(x[node_idx])
                continue
            
            valid_features = []
            current_state = go.UnifiedState(x[node_idx].cpu(), self.dimension)
            
            for neighbor_idx in neighbor_indices:
                neighbor_state = go.UnifiedState(x[neighbor_idx].cpu(), self.dimension)
                
                # === LOGIC ENGINE: Check orthogonality ===
                is_orthogonal = self.logic_engine.check_orthogonality(
                    current_state, neighbor_state
                )
                
                if is_orthogonal or neighbor_idx == node_idx:
                    # === CLIFFORD ENGINE: Apply product ===
                    if self.use_wedge:
                        result = self.clifford_bridge.wedge_product(
                            current_state, neighbor_state
                        )
                    else:
                        result = self.clifford_bridge.geometric_product(
                            current_state, neighbor_state
                        )
                    
                    valid_features.append(result.primary_data.to(self.device))
            
            if valid_features:
                aggregated = torch.stack(valid_features).mean(dim=0)
                output[node_idx] = self._apply_transformation(aggregated)
            else:
                output[node_idx] = self._apply_transformation(x[node_idx])
        
        return output
    
    def _boolean_forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Boolean forward pass (2D+ or 1D without constraints)."""
        num_nodes = x.size(0)
        output = torch.zeros_like(x)
        
        for node_idx in range(num_nodes):
            neighbor_mask = edge_index[1] == node_idx
            neighbor_indices = edge_index[0, neighbor_mask]
            
            if len(neighbor_indices) == 0:
                output[node_idx] = self._apply_transformation(x[node_idx])
                continue
            
            neighbor_features = []
            current_state = go.UnifiedState(x[node_idx].cpu(), self.dimension)
            
            for neighbor_idx in neighbor_indices:
                neighbor_state = go.UnifiedState(x[neighbor_idx].cpu(), self.dimension)
                
                # === CLIFFORD ENGINE: Apply product freely ===
                if self.use_wedge:
                    result = self.clifford_bridge.wedge_product(
                        current_state, neighbor_state
                    )
                else:
                    result = self.clifford_bridge.geometric_product(
                        current_state, neighbor_state
                    )
                
                neighbor_features.append(result.primary_data.to(self.device))
            
            aggregated = torch.stack(neighbor_features).mean(dim=0)
            output[node_idx] = self._apply_transformation(aggregated)
        
        return output
    
    def _apply_transformation(self, features: torch.Tensor) -> torch.Tensor:
        """Apply grade-preserving learnable transformation."""
        output = features.clone()
        
        for grade, blade_indices in self.grade_to_blades.items():
            if len(blade_indices) > 0:
                grade_weight = self.grade_weights[grade]
                output[blade_indices] = output[blade_indices] * grade_weight
        
        return output + self.bias
