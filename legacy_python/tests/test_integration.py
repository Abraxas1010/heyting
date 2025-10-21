"""
test_integration.py

Comprehensive integration tests for the unified system.
Tests end-to-end workflows and system coherence.
"""

import sys
from pathlib import Path
import torch

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generative_ontology import (
    UnifiedState,
    get_clifford_bridge,
    get_logic_bridge,
    get_graph_bridge,
)


class TestSystemIntegration:
    """Integration tests for complete system"""

    def test_basic_workflow(self):
        """Test basic creation and conversion workflow"""
        # Create state
        state = UnifiedState.from_vector([1, 2, 3, 4])

        # Access all views
        clifford = state.as_clifford()
        logic = state.as_logic()
        graph = state.as_graph()

        # Verify all exist
        assert clifford is not None
        assert logic is not None
        assert graph is not None

        print("✓ Basic workflow")

    def test_all_conversions(self):
        """Test all representation conversions"""
        state = UnifiedState.from_vector([1, 2, 3, 4])

        # Get bridges
        cliff_bridge = get_clifford_bridge()
        logic_bridge = get_logic_bridge()
        graph_bridge = get_graph_bridge()

        # Test all conversions work
        mv = cliff_bridge.state_to_clifford(state)
        state2 = cliff_bridge.clifford_to_state(mv, 2)

        graph = graph_bridge.state_to_graph(state)
        state3 = graph_bridge.graph_to_state(graph, 2)

        # Verify consistency
        assert torch.allclose(state.primary_data, state2.primary_data, atol=1e-10)
        assert torch.allclose(state.primary_data, state3.primary_data, atol=1e-10)

        print("✓ All conversions")

    def test_operations(self):
        """Test operations across representations"""
        state1 = UnifiedState.from_vector([0, 1, 0, 0])  # e1
        state2 = UnifiedState.from_vector([0, 0, 1, 0])  # e2

        # Clifford operations
        cliff_bridge = get_clifford_bridge()
        wedge = cliff_bridge.wedge_product(state1, state2)

        # Logic operations
        logic_bridge = get_logic_bridge()
        meet = logic_bridge.meet(state1, state2)

        # Both should produce results
        assert wedge is not None
        assert meet is not None

        print("✓ Operations")

    def test_dimension_progression(self):
        """Test progression through dimensions"""
        for dim in [1, 2, 3]:
            size = 2 ** dim
            state = UnifiedState.zero(dim)

            # Verify all views work
            clifford = state.as_clifford()
            logic = state.as_logic()
            graph = state.as_graph()

            assert graph.num_nodes == size

        print("✓ Dimension progression")

    def test_caching(self):
        """Test view caching"""
        state = UnifiedState.from_vector([1, 2, 3, 4])

        # First access
        graph1 = state.as_graph()

        # Second access (should be cached)
        graph2 = state.as_graph()

        # Should be same object
        assert graph1 is graph2

        print("✓ Caching")

    def test_round_trip_accuracy(self):
        """Test round-trip accuracy meets epsilon"""
        state = UnifiedState.from_vector([1, 2, 3, 4])

        # U → C → U
        cliff_bridge = get_clifford_bridge()
        mv = cliff_bridge.state_to_clifford(state)
        recovered = cliff_bridge.clifford_to_state(mv, 2)

        error = torch.norm(state.primary_data - recovered.primary_data).item()
        assert error < 1e-10

        print(f"✓ Round-trip accuracy (ε = {error:.2e})")

    def run_all(self):
        """Run all integration tests"""
        print("\n" + "="*60)
        print("INTEGRATION TEST SUITE")
        print("="*60)

        tests = [
            self.test_basic_workflow,
            self.test_all_conversions,
            self.test_operations,
            self.test_dimension_progression,
            self.test_caching,
            self.test_round_trip_accuracy,
        ]

        passed = 0
        failed = 0

        for test in tests:
            try:
                test()
                passed += 1
            except Exception as e:
                print(f"✗ {test.__name__}: {e}")
                failed += 1

        print("="*60)
        print(f"Results: {passed} passed, {failed} failed")
        print("="*60)

        return failed == 0


if __name__ == "__main__":
    tester = TestSystemIntegration()
    success = tester.run_all()
    sys.exit(0 if success else 1)
