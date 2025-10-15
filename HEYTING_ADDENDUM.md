# Heyting Algebra Addendum
## Adding TRUE Intuitionistic Logic via Posets

**Version**: 3.1 Addendum
**Date**: 2025-10-13
**Status**: Additions to IMPLEMENTATION_PLAN.md

---

## Research Validation

### ✅ Confirmed Claims (from web search):

1. **"Every finite Heyting algebra is the algebra of all open sets of X for some finite topological space X"** ✓
2. **"Alexandroff spaces correspond to posets where open sets are upsets (or downsets)"** ✓
3. **"Adjunction law: (c ∧ a) ≤ b iff c ≤ (a → b)"** ✓ (defining property)
4. **"Pseudocomplement: ¬a = a → ⊥"** ✓
5. **"Heyting algebras model propositional intuitionistic logic"** ✓

### Key Distinction:

- ❌ **Projectors** + Heyting = Incompatible (arXiv:1310.3604)
- ✅ **Downsets/Topology** + Heyting = Standard construction

**Conclusion**: Can add Heyting as 3rd logic type via **poset-based implementation** (not projectors).

---

## Updated Architecture: 3-Way Logic System

| Mode | Logic Type | Implementation | Key Property | Foundation |
|------|-----------|----------------|--------------|------------|
| **Heyting** | Intuitionistic | Downsets of finite poset | ¬¬a ≥ a (not equal) | Topology/Order theory |
| **Boolean** | Classical | Commuting projectors | Distributive | Operator algebra |
| **Orthomodular** | Quantum | General projectors | Non-distributive | Operator algebra |

**Mathematical Purity**: Each implemented in its PROPER framework (no cross-contamination).

---

## Part I: Heyting Algebra Implementation

### Section 1.3.5 (NEW): Heyting Algebra via Posets

**INSERT AFTER**: Section 1.3 (Lattice Operations) in main plan

#### Mathematical Foundation

**Definition**: A finite **Heyting algebra** is the lattice of downsets of a finite poset.

**Downset** (also called "order ideal"):
```
U ⊆ P is a downset iff: x ∈ U and y ≤ x ⟹ y ∈ U
```

**Equivalently**: Open sets of Alexandroff topology on P.

**Operations** (from research):
```
a ∧ b = a ∩ b                        (meet = intersection)
a ∨ b = a ∪ b                        (join = union)
a → b = {x ∈ P | ↑x ∩ a ⊆ b}        (implication)
¬a = a → ⊥ = {x ∈ P | ↑x ∩ a = ∅}  (negation)
```

where `↑x = {y ∈ P | x ≤ y}` (principal upset)

**Key Property**:
- **Adjunction**: `c ≤ (a → b)` iff `(c ∧ a) ≤ b` (defining property of Heyting algebras)
- **Double negation**: `a ≤ ¬¬a` but NOT necessarily `a = ¬¬a` (intuitionistic)
- **Excluded middle**: `a ∨ ¬a ≠ ⊤` in general (fails for non-Boolean posets)

---

#### Implementation: Bitset Representation

**File**: `logic/heyting_poset.py` (NEW)

```python
"""
Heyting algebra via downsets of finite poset.

Research foundation:
- "Every finite Heyting algebra is isomorphic to the lattice of
   downsets of some finite poset" (Birkhoff's theorem)
- Alexandroff topology: Open sets = downsets (or upsets, dual convention)
"""

import numpy as np
import torch
from typing import List, Set, Tuple, Dict


class Poset:
    """
    Finite partially ordered set (poset).

    Representation:
    - Elements: 0, 1, ..., n-1
    - Order relation: stored as reachability matrix R
      R[i,j] = True iff i ≤ j
    """

    def __init__(self, n: int):
        """
        Initialize poset with n elements.

        Args:
            n: Number of elements
        """
        self.n = n
        # Reachability matrix (transitive closure of order)
        self.R = np.eye(n, dtype=bool)  # Reflexive by default

    def add_edge(self, u: int, v: int):
        """
        Add order relation u ≤ v.

        Automatically computes transitive closure.
        """
        if u < 0 or u >= self.n or v < 0 or v >= self.n:
            raise ValueError(f"Invalid elements: {u}, {v}")

        self.R[u, v] = True
        self._transitive_closure()

    def _transitive_closure(self):
        """
        Compute transitive closure via Floyd-Warshall.

        After this, R[i,j] = True iff i ≤ j.
        """
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if self.R[i, k] and self.R[k, j]:
                        self.R[i, j] = True

    def up(self, x: int) -> np.ndarray:
        """
        Principal upset: ↑x = {y | x ≤ y}

        Returns: Boolean mask of elements in ↑x
        """
        return self.R[x, :]

    def down(self, x: int) -> np.ndarray:
        """
        Principal downset: ↓x = {y | y ≤ x}

        Returns: Boolean mask of elements in ↓x
        """
        return self.R[:, x]

    def is_downset(self, U: np.ndarray) -> bool:
        """
        Check if U is a downset.

        Downset property: x ∈ U and y ≤ x ⟹ y ∈ U

        Args:
            U: Boolean mask (size n)
        """
        if U.shape[0] != self.n:
            raise ValueError(f"Expected array of size {self.n}, got {U.shape[0]}")

        for x in range(self.n):
            if U[x]:  # x is in U
                # Check all y ≤ x
                below_x = self.down(x)
                # All elements below x must be in U
                if not np.all(U[below_x]):
                    return False

        return True

    def normalize_to_downset(self, U: np.ndarray) -> np.ndarray:
        """
        Convert arbitrary set to smallest downset containing it.

        Algorithm: For each x in U, add all y ≤ x.
        """
        result = U.copy()
        for x in range(self.n):
            if U[x]:
                # Add all elements below x
                result = result | self.down(x)
        return result


class HeytingPoset:
    """
    Heyting algebra as lattice of downsets of a poset.

    Elements: Downsets of P (represented as bitsets/boolean arrays)
    Operations: Meet, join, implication, negation

    Mathematical properties:
    - Distributive lattice (finite case)
    - Implication via adjunction: c ≤ (a→b) iff (c∧a) ≤ b
    - Pseudocomplement: ¬a = a → ⊥
    - Double negation: a ≤ ¬¬a (but not necessarily equal)
    """

    def __init__(self, poset: Poset):
        """
        Initialize Heyting algebra from poset.

        Args:
            poset: Underlying finite poset
        """
        self.poset = poset
        self.n = poset.n
        self.logic_type = "heyting"

        # Precompute upsets for efficiency
        self._upsets = np.array([poset.up(x) for x in range(self.n)])

    def top(self) -> np.ndarray:
        """Top element: ⊤ = all elements"""
        return np.ones(self.n, dtype=bool)

    def bottom(self) -> np.ndarray:
        """Bottom element: ⊥ = empty set"""
        return np.zeros(self.n, dtype=bool)

    def meet(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Meet: a ∧ b = a ∩ b

        Intersection of downsets is a downset.
        """
        return a & b

    def join(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Join: a ∨ b = a ∪ b

        Union of downsets is a downset.
        """
        return a | b

    def implies(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Implication: a → b

        Definition: {x ∈ P | ↑x ∩ a ⊆ b}

        Algorithm (bitset):
        For each x, check if (↑x & a) ⊆ b
        Equivalently: (↑x & a & ~b) == ∅
        """
        result = np.zeros(self.n, dtype=bool)

        for x in range(self.n):
            up_x = self._upsets[x]
            # Check if (↑x ∩ a) ⊆ b
            # i.e., (↑x & a) - b = ∅
            if not np.any(up_x & a & ~b):
                result[x] = True

        # Normalize to downset (should already be one, but verify)
        return self.poset.normalize_to_downset(result)

    def negate(self, a: np.ndarray) -> np.ndarray:
        """
        Negation: ¬a = a → ⊥

        Definition: {x ∈ P | ↑x ∩ a = ∅}

        This is the pseudocomplement.
        """
        return self.implies(a, self.bottom())

    def leq(self, a: np.ndarray, b: np.ndarray) -> bool:
        """
        Check order: a ≤ b

        For downsets: a ≤ b iff a ⊆ b
        """
        return np.all(a <= b)  # Element-wise comparison

    # === Verification ===

    def verify_heyting_axioms(self) -> Dict[str, bool]:
        """
        Verify all Heyting algebra axioms.

        Must pass:
        1. Distributivity: a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)
        2. Adjunction: c ≤ (a → b) iff (c ∧ a) ≤ b
        3. Double negation: a ≤ ¬¬a

        May fail (intuitionistic properties):
        4. Excluded middle: a ∨ ¬a = ⊤ (fails for non-Boolean)
        5. Double negation law: ¬¬a = a (fails)
        """
        results = {}

        # Generate random downsets for testing
        a = self._random_downset()
        b = self._random_downset()
        c = self._random_downset()

        # 1. Distributivity
        lhs = self.meet(a, self.join(b, c))
        rhs = self.join(self.meet(a, b), self.meet(a, c))
        results['distributive'] = np.array_equal(lhs, rhs)

        # 2. Adjunction law (defining property)
        a_imp_b = self.implies(a, b)

        # Forward: c ≤ (a → b) ⟹ (c ∧ a) ≤ b
        if self.leq(c, a_imp_b):
            c_meet_a = self.meet(c, a)
            results['adjunction_forward'] = self.leq(c_meet_a, b)
        else:
            results['adjunction_forward'] = True  # Vacuously true

        # Reverse: (c ∧ a) ≤ b ⟹ c ≤ (a → b)
        c_meet_a = self.meet(c, a)
        if self.leq(c_meet_a, b):
            results['adjunction_reverse'] = self.leq(c, a_imp_b)
        else:
            results['adjunction_reverse'] = True  # Vacuously true

        # 3. Double negation: a ≤ ¬¬a
        not_a = self.negate(a)
        not_not_a = self.negate(not_a)
        results['double_negation_inequality'] = self.leq(a, not_not_a)

        # 4. Excluded middle (should FAIL for non-Boolean)
        a_or_not_a = self.join(a, self.negate(a))
        top = self.top()
        results['excluded_middle'] = np.array_equal(a_or_not_a, top)

        # 5. Double negation law (should FAIL)
        results['double_negation_equality'] = np.array_equal(a, not_not_a)

        return results

    def _random_downset(self) -> np.ndarray:
        """Generate random downset for testing"""
        # Random subset
        subset = np.random.rand(self.n) > 0.5
        # Normalize to downset
        return self.poset.normalize_to_downset(subset)

    # === Example Posets ===

    @staticmethod
    def diamond() -> 'HeytingPoset':
        """
        Create diamond poset (non-distributive, non-Boolean).

        Structure:
            ⊤
           / \
          a   b
           \ /
            ⊥

        This has 4 elements: ⊥ < a < ⊤, ⊥ < b < ⊤
        Downsets: {∅, {⊥}, {⊥,a}, {⊥,b}, {⊥,a,b,⊤}}
        """
        poset = Poset(4)
        # Order: 0=⊥, 1=a, 2=b, 3=⊤
        poset.add_edge(0, 1)  # ⊥ ≤ a
        poset.add_edge(0, 2)  # ⊥ ≤ b
        poset.add_edge(1, 3)  # a ≤ ⊤
        poset.add_edge(2, 3)  # b ≤ ⊤
        return HeytingPoset(poset)

    @staticmethod
    def chain(n: int) -> 'HeytingPoset':
        """
        Create chain poset: 0 < 1 < 2 < ... < n-1

        This gives a Boolean algebra (totally ordered).
        """
        poset = Poset(n)
        for i in range(n - 1):
            poset.add_edge(i, i + 1)
        return HeytingPoset(poset)
```

**Success Criteria**:
- [ ] Distributivity verified (all finite Heyting algebras are distributive)
- [ ] Adjunction law holds: `c ≤ (a → b)` iff `(c ∧ a) ≤ b`
- [ ] Double negation: `a ≤ ¬¬a` (but not equality)
- [ ] Excluded middle FAILS for non-Boolean posets (diamond example)
- [ ] Implication algorithm correct (bitset implementation)

---

### Section 1.3.6 (NEW): Heyting ↔ Other Structures Bridges

**INSERT AFTER**: Section 1.3.5

#### Heyting ↔ Graph Bridge

**Representation**: Poset as Directed Acyclic Graph (DAG)

**Graph Structure**:
```
Nodes: Elements of poset P
Edges: (u, v) iff u < v in Hasse diagram (cover relations only)
Node features: Downset membership (bitset)
```

**Bridge Operations**:
```python
def heyting_to_graph(downset: np.ndarray, poset: Poset) -> PyG.Data:
    """
    Convert Heyting element (downset) to graph.

    Returns:
    - x: Node features (downset membership for this element)
    - edge_index: Hasse diagram (cover relations)
    """
    # Hasse diagram edges (cover relations)
    edge_list = []
    for u in range(poset.n):
        for v in range(poset.n):
            if poset.R[u, v] and u != v:
                # Check if cover (no element strictly between)
                is_cover = True
                for w in range(poset.n):
                    if w != u and w != v:
                        if poset.R[u, w] and poset.R[w, v]:
                            is_cover = False
                            break
                if is_cover:
                    edge_list.append([u, v])

    edge_index = torch.tensor(edge_list, dtype=torch.long).T

    # Node features: membership in this downset
    x = torch.tensor(downset, dtype=torch.float32).unsqueeze(1)

    return PyG.Data(x=x, edge_index=edge_index)
```

**Key Insight**: Heyting algebra operations become **graph operations on DAGs**.

---

#### Heyting ↔ Tensor Bridge

**Representation**: Downset as Boolean vector

```python
def heyting_to_tensor(downset: np.ndarray) -> torch.Tensor:
    """
    Convert downset to tensor.

    Simply: Boolean mask → float tensor
    """
    return torch.tensor(downset, dtype=torch.float32)

def tensor_to_heyting(tensor: torch.Tensor, poset: Poset) -> np.ndarray:
    """
    Convert tensor to downset.

    Algorithm:
    1. Threshold to boolean
    2. Normalize to downset
    """
    boolean_mask = (tensor > 0.5).cpu().numpy()
    return poset.normalize_to_downset(boolean_mask)
```

---

#### Heyting ↔ Clifford: NO Direct Bridge

**IMPORTANT**: Heyting algebra (topology/posets) and Clifford algebra (geometric) are **separate mathematical structures**.

**DO NOT** force algebraic isomorphism. Instead:

1. **Option A**: Treat as separate reasoning channels
   - Heyting: Symbolic/logical reasoning
   - Clifford: Geometric reasoning
   - Coordinate via **consistency losses** (learned alignment)

2. **Option B**: Map via intermediate representation
   - Heyting downset → Graph → Clifford multivector
   - Indirect bridge through shared graph structure

**Research Note**: No standard mathematical correspondence between Heyting algebras and Clifford algebras. They model different phenomena.

---

## Part II: Updated Factory & Modes

### Update LogicEngine Factory (Section 2.3)

**CHANGE**: Add Heyting mode (3-way dispatch)

```python
class LogicEngine:
    """
    Factory for three logic types:
    - 'heyting': Intuitionistic logic (downsets of poset)
    - 'boolean': Classical logic (commuting projectors)
    - 'orthomodular': Quantum logic (general projectors)
    """

    def __init__(self, mode: str, dimension: int = 4, poset: Poset = None):
        """
        Initialize logic engine.

        Args:
            mode: 'heyting', 'boolean', or 'orthomodular'
            dimension: Matrix dimension (for projector modes)
            poset: Poset for Heyting mode (default: chain of size dimension)
        """
        if mode not in ['heyting', 'boolean', 'orthomodular']:
            raise ValueError(f"Mode must be 'heyting', 'boolean', or 'orthomodular'")

        self.mode = mode
        self.dimension = dimension

        # Instantiate backend
        if mode == 'heyting':
            if poset is None:
                # Default: chain poset (gives Boolean algebra)
                # For TRUE Heyting, use diamond or custom poset
                poset = Poset(dimension)
                for i in range(dimension - 1):
                    poset.add_edge(i, i + 1)

            self.backend = HeytingPoset(poset)
            self.poset = poset

        elif mode == 'boolean':
            self.backend = BooleanMode()

        elif mode == 'orthomodular':
            self.backend = OrthomodularMode()

    # === Type Checking ===

    def is_heyting(self) -> bool:
        """Check if using Heyting (intuitionistic) logic"""
        return self.mode == 'heyting'

    def is_boolean(self) -> bool:
        """Check if using Boolean (classical) logic"""
        return self.mode == 'boolean'

    def is_orthomodular(self) -> bool:
        """Check if using orthomodular (quantum) logic"""
        return self.mode == 'orthomodular'

    # === Core Operations ===

    def meet(self, a, b):
        """Logical AND: a ∧ b"""
        return self.backend.meet(a, b)

    def join(self, a, b):
        """Logical OR: a ∨ b"""
        return self.backend.join(a, b)

    def negate(self, a):
        """Logical NOT: ¬a"""
        if self.is_heyting():
            return self.backend.negate(a)
        else:
            return self.backend.complement(a)

    def implies(self, a, b):
        """Logical implication: a → b"""
        return self.backend.implies(a, b)

    # === Mode-Specific Operations ===

    def check_excluded_middle(self, a) -> bool:
        """
        Check if excluded middle holds: a ∨ ¬a = ⊤

        Expected:
        - Heyting: May FAIL (intuitionistic property)
        - Boolean: Always True
        - Orthomodular: Always True
        """
        not_a = self.negate(a)
        a_or_not_a = self.join(a, not_a)

        if self.is_heyting():
            top = self.backend.top()
            return np.array_equal(a_or_not_a, top)
        else:
            I = np.eye(self.dimension)
            return np.allclose(a_or_not_a.data, I, atol=1e-10)

    def check_double_negation(self, a) -> bool:
        """
        Check if double negation law holds: ¬¬a = a

        Expected:
        - Heyting: FAILS (¬¬a ≥ a, but not equal)
        - Boolean: Always True
        - Orthomodular: Always True
        """
        not_not_a = self.negate(self.negate(a))

        if self.is_heyting():
            return np.array_equal(a, not_not_a)
        else:
            return np.allclose(a.data, not_not_a.data, atol=1e-10)
```

**Success Criteria**:
- [ ] 3-way dispatch works
- [ ] Heyting mode uses poset backend
- [ ] Boolean/Orthomodular unchanged
- [ ] Type checking methods work

---

## Part III: Neural Architecture Updates

### Update GeometricMessagePassing (Task 3.1)

**CHANGE**: 3-way logic dispatch

```python
# === LOGIC ENGINE: 3-way mode-dependent constraints ===

if self.logic_engine.is_heyting():
    # Intuitionistic: Use poset order constraints
    # Check if elements are comparable in poset
    state_j_downset = extract_downset(state_j)  # Helper function
    state_i_downset = extract_downset(state_i)

    # Only process if elements are in compatible downsets
    # (implementation detail: define compatibility via poset order)
    if not compatible_in_poset(state_j_downset, state_i_downset, self.logic_engine.poset):
        continue  # Skip incompatible pairs

    message_weight = 1.0  # No weighting needed

elif self.logic_engine.is_orthomodular():
    # Quantum: Compatibility weighting
    is_compatible = self.logic_engine.commute(state_j, state_i)

    if not is_compatible:
        comm = self.logic_engine.commutator(state_j, state_i)
        comm_magnitude = comm.norm(ord='fro')
        compatibility_weight = 1.0 / (1.0 + comm_magnitude)
    else:
        compatibility_weight = 1.0

elif self.logic_engine.is_boolean():
    # Classical: No constraints
    compatibility_weight = 1.0
```

---

### Update LogicAwareConv (Task 3.2)

**CHANGE**: Add `_heyting_forward` method

```python
def forward(self, x, edge_index, apply_constraints=True):
    """3-way dispatch based on logic mode"""
    if self.logic_engine.is_heyting() and apply_constraints:
        return self._heyting_forward(x, edge_index)
    elif self.logic_engine.is_orthomodular() and apply_constraints:
        return self._orthomodular_forward(x, edge_index)
    else:
        return self._boolean_forward(x, edge_index)

def _heyting_forward(self, x, edge_index):
    """
    Heyting-constrained forward pass.

    Uses poset order to filter operations.
    """
    num_nodes = x.size(0)
    output = torch.zeros_like(x)

    for node_idx in range(num_nodes):
        neighbor_mask = edge_index[1] == node_idx
        neighbor_indices = edge_index[0, neighbor_mask]

        if len(neighbor_indices) == 0:
            output[node_idx] = self._apply_transformation(x[node_idx])
            continue

        # Convert to downsets
        current_downset = self._to_downset(x[node_idx])

        valid_features = []
        for neighbor_idx in neighbor_indices:
            neighbor_downset = self._to_downset(x[neighbor_idx])

            # === HEYTING LOGIC: Check poset compatibility ===
            # (define compatibility via order relation)

            # Apply logical operations (meet/join/implies)
            result = self.logic_engine.implies(current_downset, neighbor_downset)

            valid_features.append(self._from_downset(result))

        if valid_features:
            aggregated = torch.stack(valid_features).mean(dim=0)
            output[node_idx] = self._apply_transformation(aggregated)
        else:
            output[node_idx] = self._apply_transformation(x[node_idx])

    return output

def _to_downset(self, features: torch.Tensor) -> np.ndarray:
    """Convert tensor features to downset bitset"""
    return (features.cpu().numpy() > 0.5).astype(bool)

def _from_downset(self, downset: np.ndarray) -> torch.Tensor:
    """Convert downset bitset to tensor features"""
    return torch.tensor(downset, dtype=torch.float32, device=self.device)
```

---

## Part IV: Testing

### Add Heyting Unit Tests

**File**: `tests/test_heyting_poset.py` (NEW)

```python
"""
Unit tests for Heyting algebra via posets.
"""

import pytest
import numpy as np
from logic.heyting_poset import Poset, HeytingPoset


class TestPoset:
    def test_chain(self):
        """Test chain poset: 0 < 1 < 2"""
        poset = Poset(3)
        poset.add_edge(0, 1)
        poset.add_edge(1, 2)

        # Verify transitivity: 0 < 2
        assert poset.R[0, 2]

    def test_diamond(self):
        """Test diamond poset (non-Boolean)"""
        poset = Poset(4)
        # 0 < 1, 0 < 2, 1 < 3, 2 < 3
        poset.add_edge(0, 1)
        poset.add_edge(0, 2)
        poset.add_edge(1, 3)
        poset.add_edge(2, 3)

        # Verify: 1 and 2 are incomparable
        assert not poset.R[1, 2]
        assert not poset.R[2, 1]

        # Verify: 0 < 3 (transitivity)
        assert poset.R[0, 3]

    def test_downset_check(self):
        """Test downset verification"""
        poset = Poset(3)
        poset.add_edge(0, 1)
        poset.add_edge(1, 2)

        # {0} is a downset
        assert poset.is_downset(np.array([True, False, False]))

        # {0, 1} is a downset
        assert poset.is_downset(np.array([True, True, False]))

        # {1} is NOT a downset (missing 0 below it)
        assert not poset.is_downset(np.array([False, True, False]))

    def test_normalize_downset(self):
        """Test downset normalization"""
        poset = Poset(3)
        poset.add_edge(0, 1)
        poset.add_edge(1, 2)

        # Start with {2} (not a downset)
        subset = np.array([False, False, True])

        # Normalize: should become {0, 1, 2}
        normalized = poset.normalize_to_downset(subset)
        expected = np.array([True, True, True])

        assert np.array_equal(normalized, expected)


class TestHeytingPoset:
    def setup_method(self):
        """Set up test fixtures"""
        # Diamond poset (non-Boolean)
        self.heyting = HeytingPoset.diamond()

    def test_distributivity(self):
        """Test: a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)"""
        a = np.array([True, True, False, False])   # {⊥, a}
        b = np.array([True, False, True, False])   # {⊥, b}
        c = np.array([True, False, False, False])  # {⊥}

        lhs = self.heyting.meet(a, self.heyting.join(b, c))
        rhs = self.heyting.join(
            self.heyting.meet(a, b),
            self.heyting.meet(a, c)
        )

        assert np.array_equal(lhs, rhs), "Distributivity must hold"

    def test_adjunction_law(self):
        """Test: c ≤ (a → b) iff (c ∧ a) ≤ b"""
        a = np.array([True, True, False, False])
        b = np.array([True, False, True, False])
        c = np.array([True, False, False, False])

        # Compute a → b
        a_imp_b = self.heyting.implies(a, b)

        # Forward: c ≤ (a → b) ⟹ (c ∧ a) ≤ b
        if self.heyting.leq(c, a_imp_b):
            c_meet_a = self.heyting.meet(c, a)
            assert self.heyting.leq(c_meet_a, b), "Adjunction forward direction"

        # Reverse: (c ∧ a) ≤ b ⟹ c ≤ (a → b)
        c_meet_a = self.heyting.meet(c, a)
        if self.heyting.leq(c_meet_a, b):
            assert self.heyting.leq(c, a_imp_b), "Adjunction reverse direction"

    def test_double_negation_inequality(self):
        """Test: a ≤ ¬¬a (but not necessarily equal)"""
        a = np.array([True, True, False, False])

        not_a = self.heyting.negate(a)
        not_not_a = self.heyting.negate(not_a)

        # Must have a ≤ ¬¬a
        assert self.heyting.leq(a, not_not_a), "Double negation inequality"

        # For diamond, should NOT have equality (intuitionistic property)
        # (This depends on specific downset, but common case)

    def test_excluded_middle_fails(self):
        """Test: a ∨ ¬a ≠ ⊤ for some a (intuitionistic property)"""
        # For diamond, try a = {⊥, a}
        a = np.array([True, True, False, False])

        not_a = self.heyting.negate(a)
        a_or_not_a = self.heyting.join(a, not_a)
        top = self.heyting.top()

        # Excluded middle should FAIL for diamond
        assert not np.array_equal(a_or_not_a, top), \
            "Excluded middle should fail for non-Boolean poset"

    def test_chain_is_boolean(self):
        """Test: Chain poset gives Boolean algebra"""
        chain = HeytingPoset.chain(4)

        # For any downset in chain, excluded middle should hold
        a = np.array([True, True, False, False])  # First two elements

        not_a = chain.negate(a)
        a_or_not_a = chain.join(a, not_a)
        top = chain.top()

        # For chain (totally ordered), Heyting = Boolean
        assert np.array_equal(a_or_not_a, top), \
            "Chain should satisfy excluded middle"
```

**Success Criteria**:
- [ ] All Heyting axioms verified
- [ ] Adjunction law holds (defining property)
- [ ] Excluded middle FAILS for diamond (non-Boolean)
- [ ] Chain poset gives Boolean algebra (as expected)

---

## Part V: Updated Timeline

### Phase 0: Heyting Implementation (NEW) - 4-5 hours

0.1. Create `Poset` class (1 hour)
0.2. Create `HeytingPoset` class (2 hours)
0.3. Implement implication algorithm (bitset) (1 hour)
0.4. Write Heyting unit tests (1-2 hours)

### Original Phases (unchanged times)

**Total NEW time**: +4-5 hours
**Revised TOTAL**: 25-33 hours (was 21-28)

---

## Part VI: Updated Architecture Table

### Final 3-Way System

| Property | Heyting | Boolean | Orthomodular |
|----------|---------|---------|--------------|
| **Implementation** | Downsets (poset) | Commuting projectors | General projectors |
| **Foundation** | Topology/Order | Operator algebra | Operator algebra |
| **Distributive** | YES | YES | NO |
| **Involution** | NO (¬¬a ≥ a) | YES (¬¬a = a) | YES (⊥⊥P = P) |
| **Excluded middle** | FAILS | Holds | Holds |
| **Commutativity** | N/A | Required | Optional |
| **Models** | Intuitionistic | Classical | Quantum |
| **Applications** | Constructive logic | Boolean circuits | Quantum computing |

---

## Part VII: Bridge Strategy

### Three Separate Frameworks

```
Heyting (Topology)          Boolean (Commuting)       Orthomodular (Non-commuting)
     ↓                              ↓                           ↓
 Downsets                      Diagonal                    Matrix
  (bitsets)                   Projectors                 Projectors
     ↓                              ↓                           ↓
  Poset DAG  ←──[Graph]──→  Compatibility  ←──[Graph]──→  Compatibility
                              Graph                        Graph
```

**NO direct Heyting ↔ Projector bridge** (mathematically distinct structures)

**Coordination**: Via shared Graph representation + learned consistency losses

---

## Part VIII: Research Citations (Updated)

### Heyting Algebra Sources

1. **Birkhoff's Representation Theorem**: "Every finite Heyting algebra is isomorphic to the lattice of downsets of some finite poset"

2. **Alexandroff Topology**: "Finite topological spaces ↔ Preorders (open sets = upsets/downsets)"

3. **Adjunction Property**: "(c ∧ a) ≤ b iff c ≤ (a → b)" - defining property of Heyting algebras

4. **arXiv:1310.3604**: Confirms projectors incompatible with Heyting (but topology approach is valid)

---

## Document Metadata

**Version**: 3.1 Addendum
**Created**: 2025-10-13
**Author**: Claude Code
**Status**: Ready to Merge into IMPLEMENTATION_PLAN.md

**To Merge**: Insert sections at appropriate locations in main plan, update all "2-way" → "3-way".

---

*END OF ADDENDUM*
