"""
reasoning_engines.py

Core reasoning mode engines: Induction, Deduction, Abduction
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
from enum import Enum


class Representation(Enum):
    """The four sheaves/representations"""
    UNIFIED = "U"
    CLIFFORD = "C"
    LOGIC = "L"
    GRAPH = "G"


class ReasoningMode(Enum):
    """The three reasoning operations"""
    INDUCTION = "I"
    DEDUCTION = "D"
    ABDUCTION = "A"


class ReasoningEngine(ABC):
    """Base class for all reasoning engines"""

    @abstractmethod
    def apply(self, view: Any, representation: Representation,
              context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        pass


class InductionEngine(ReasoningEngine):
    """
    (Data + Answer) → Rules
    Learn patterns from examples.
    """

    def __init__(self):
        self.learned_models = {}

    def apply(self, view: Any, representation: Representation,
              context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        samples = context.get('training_samples', [])
        if not samples:
            return view, context

        epochs = context.get('epochs', 50)
        lr = context.get('learning_rate', 1e-3)

        model_key = f"{representation.value}_model"
        if model_key not in context:
            context[model_key] = self._create_model(representation)

        model = context[model_key]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0
            for sample in samples:
                data = sample.get('data', sample.get('input'))
                answer = sample.get('answer', sample.get('output'))

                if representation == Representation.GRAPH:
                    prediction = model(data)
                else:
                    data_tensor = self._to_tensor(data, representation)
                    prediction = model(data_tensor)

                target_tensor = self._to_tensor(answer, representation)
                loss = nn.functional.mse_loss(prediction, target_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0 and context.get('verbose', False):
                print(f"  Induction epoch {epoch}: loss={total_loss:.6f}")

        self.learned_models[representation.value] = model
        context['learned_model'] = model
        return view, context

    def _create_model(self, representation: Representation) -> nn.Module:
        if representation == Representation.GRAPH:
            from torch_geometric.nn import GCNConv

            class SimpleGNN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = GCNConv(3, 16)
                    self.conv2 = GCNConv(16, 3)

                def forward(self, data):
                    x, edge_index = data.x, data.edge_index
                    x = self.conv1(x, edge_index).relu()
                    x = self.conv2(x, edge_index)
                    return x

            return SimpleGNN()
        else:
            class SimpleMLP(nn.Module):
                def __init__(self, input_dim=4):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, 32),
                        nn.ReLU(),
                        nn.Linear(32, 32),
                        nn.ReLU(),
                        nn.Linear(32, input_dim)
                    )

                def forward(self, x):
                    return self.net(x)

            return SimpleMLP()

    def _to_tensor(self, obj: Any, representation: Representation) -> torch.Tensor:
        if isinstance(obj, torch.Tensor):
            return obj
        if hasattr(obj, 'primary_data'):
            return obj.primary_data
        return torch.tensor(obj, dtype=torch.float32)


class DeductionEngine(ReasoningEngine):
    """
    (Rules + Data) → Answer
    Apply known rules to derive answers.
    """

    def apply(self, view: Any, representation: Representation,
              context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        model = context.get('learned_model')

        if model is not None:
            with torch.no_grad():
                if representation == Representation.GRAPH:
                    result = model(view)
                else:
                    input_tensor = self._to_tensor(view, representation)
                    result = model(input_tensor)
            context['deduction_result'] = result
            return result, context

        operation = context.get('operation')
        if operation is None:
            return view, context

        result = self._apply_geometric_operation(view, representation,
                                                  operation, context)
        context['deduction_result'] = result
        return result, context

    def _apply_geometric_operation(self, view: Any, representation: Representation,
                                   operation: str, context: Dict[str, Any]) -> Any:
        from bridges.logic_clifford import get_clifford_bridge, get_logic_bridge
        from core.unified_state import UnifiedState

        if isinstance(view, UnifiedState):
            state = view
        else:
            state = view

        operand_b = context.get('operand_b')

        if operation == 'wedge' and operand_b is not None:
            return get_clifford_bridge().wedge_product(state, operand_b)
        elif operation == 'inner' and operand_b is not None:
            return get_clifford_bridge().inner_product(state, operand_b)
        elif operation == 'negate':
            return get_logic_bridge().negate(state)
        elif operation == 'meet' and operand_b is not None:
            return get_logic_bridge().meet(state, operand_b)
        return view

    def _to_tensor(self, obj: Any, representation: Representation) -> torch.Tensor:
        if isinstance(obj, torch.Tensor):
            return obj
        if hasattr(obj, 'primary_data'):
            return obj.primary_data
        return torch.tensor(obj, dtype=torch.float32)


class AbductionEngine(ReasoningEngine):
    """
    (Rules + Answer) → Data
    Generate plausible inputs for desired output.
    """

    def apply(self, view: Any, representation: Representation,
              context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        target = context.get('target')
        if target is None:
            return view, context

        max_iters = context.get('max_iterations', 500)
        lr = context.get('learning_rate', 1e-2)
        dimension = context.get('dimension', 2)
        size = 2 ** dimension

        generated_input = torch.randn(size, requires_grad=True)
        optimizer = torch.optim.Adam([generated_input], lr=lr)

        model = context.get('learned_model')
        operation = context.get('operation')

        for iteration in range(max_iters):
            optimizer.zero_grad()

            if model is not None:
                prediction = model(generated_input)
            elif operation:
                from core.unified_state import UnifiedState
                state = UnifiedState(generated_input, dimension)
                prediction = self._apply_operation(state, operation, context)
                if hasattr(prediction, 'primary_data'):
                    prediction = prediction.primary_data
            else:
                prediction = generated_input

            target_tensor = self._to_tensor(target, representation)
            loss = nn.functional.mse_loss(prediction, target_tensor)

            loss.backward()
            optimizer.step()

            if iteration % 100 == 0 and context.get('verbose', False):
                print(f"  Abduction iter {iteration}: loss={loss.item():.6f}")

            if loss.item() < 1e-8:
                break

        from core.unified_state import UnifiedState
        generated_state = UnifiedState(generated_input.detach(), dimension)

        context['generated_input'] = generated_state
        context['abduction_loss'] = loss.item()
        return generated_state, context

    def _apply_operation(self, state, operation, context):
        operand_b = context.get('operand_b')
        from bridges.logic_clifford import get_clifford_bridge
        bridge = get_clifford_bridge()

        if operation == 'wedge' and operand_b is not None:
            return bridge.wedge_product(state, operand_b)
        elif operation == 'inner' and operand_b is not None:
            return bridge.inner_product(state, operand_b)
        return state

    def _to_tensor(self, obj: Any, representation: Representation) -> torch.Tensor:
        if isinstance(obj, torch.Tensor):
            return obj
        if hasattr(obj, 'primary_data'):
            return obj.primary_data
        return torch.tensor(obj, dtype=torch.float32)
