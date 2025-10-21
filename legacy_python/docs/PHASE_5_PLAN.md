# Plan For Task 18+: Unified Generative Ontology System
## Remaining Implementation Plan (Phase 4+)

**Status**: Phases 1-3 Complete (78 tests passing)  
**Current**: Phase 4 Foundation Complete (Combinatorial engine operational)  
**Remaining**: Phases 5-11 (Neural architecture, applications, optimization)

---

## Core Principle: Tripartite Engine Utilization

Every task must explicitly use **all three engines**:

### 🔷 Logic Engine (Symbolic Reasoning)
- **Heyting (1D)**: Constraints on operations, excluded middle may fail
- **Boolean (2D+)**: Classical logic, all operations defined
- **Operations**: meet ∧, join ∨, negate ¬, implies →
- **Learning Role**: Learn WHEN operations are valid, constraint discovery

### 🔶 Clifford Engine (Geometric Operations)  
- **Grades**: 0 (scalar), 1 (vector), 2 (bivector), 3 (trivector)
- **Operations**: wedge ∧, inner ·, geometric product, rotors
- **Learning Role**: Learn HOW to compute geometric transformations

### 🔵 Graph Engine (Neural Learning)
- **Structure**: Nodes = blades, Edges = geometric products
- **Operations**: Message passing, aggregation, pooling
- **Learning Role**: Learn PATTERNS from data, generalization

---

## Phase 5: Neural Architecture (Critical Path)

**Goal**: Build GNN layers that respect all three algebraic structures

---

### Task 18: Geometric Message Passing Layer

**File**: `graph/layers.py`

**Purpose**: Create message passing that preserves Clifford structure AND respects Logic constraints

**Engine Integration**:
```python
class GeometricMessagePassing(MessagePassing):
    """
    Message passing respecting all three algebras.
    
    Clifford: Messages preserve grade structure
    Logic: Check if operations are defined before passing
    Graph: Standard PyG message passing framework
    """
    
    def message(self, x_j, edge_attr):
        """
        Generate messages using geometric product.
        
        Uses: CliffordEngine.geometric_product
        Constraint: LogicEngine.check_orthogonality (if 1D)
        """
        # Stub:
        # geometric_result = clifford_engine.geometric_product(x_j, edge_attr)
        # if dimension == 1:
        #     if not logic_engine.check_orthogonality(x_j, edge_attr):
        #         return zero_message  # Operation not defined
        # return geometric_result
        
    def aggregate(self, messages, index):
        """
        Aggregate respecting grade structure.
        
        Uses: Clifford grade projection to separate components
        Constraint: Logic meet operation (aggregate = meet of messages)
        """
        # Stub:
        # Separate messages by grade using clifford_engine.grade_project
        # For each grade, aggregate separately
        # Combine using logic_engine.join (OR operation)
        
    def update(self, aggr_out, x):
        """
        Update nodes using logical implication.
        
        Uses: LogicEngine.implies for state update
        """
        # Stub:
        # new_state = logic_engine.implies(x, aggr_out)
        # Interpretation: x → aggr_out (current implies new)
```

**Success Criteria**:
- ✓ Geometric product commutes with message passing
- ✓ Grade preservation: grade(message(a,b)) = grade(a) + grade(b)
- ✓ Respects 1D constraints: messages zero when non-orthogonal
- ✓ Round-trip: state → graph → message → state (ε < 10⁻⁸)

**Testing Strategy**:
1. **Clifford Test**: Verify (a∧b) passes correctly through layer
2. **Logic Test**: Verify 1D constraints respected, 2D always passes
3. **Graph Test**: Verify PyG compatibility, batching works

---

### Task 19: Logic-Aware Convolution Layer

**File**: `graph/layers.py` (append)

**Purpose**: Convolution that switches behavior based on logic type (Heyting vs Boolean)

**Engine Integration**:
```python
class LogicAwareConv(nn.Module):
    """
    Convolution with dimension-dependent logic.
    
    1D (Heyting): Partial operations, constraint checking
    2D+ (Boolean): Full operations, no constraints
    
    All dimensions: Equivariant to Clifford rotations
    """
    
    def __init__(self, dimension):
        """
        Initialize with logic type awareness.
        
        Sets up:
        - Logic engine for this dimension
        - Clifford engine for geometric operations
        - Separate processing paths for Heyting vs Boolean
        """
        self.logic_engine = LogicEngine(dimension)
        self.clifford_engine = CliffordEngine(dimension)
        self.is_heyting = (dimension == 1)
        
    def forward(self, x, edge_index):
        """
        Forward pass with logic-dependent behavior.
        
        Uses:
        1. LogicEngine: Check operation validity
        2. CliffordEngine: Apply geometric transformations
        3. GraphEngine: Structure access
        """
        # Stub:
        # if self.is_heyting:
        #     # Check orthogonality before operations
        #     valid_mask = logic_engine.check_orthogonality(x_i, x_j)
        #     # Only process valid pairs
        #     result = clifford_engine.wedge_product(x_i[valid_mask], x_j[valid_mask])
        # else:
        #     # All operations valid in Boolean
        #     result = clifford_engine.wedge_product(x_i, x_j)
        
    def verify_equivariance(self, x, rotor):
        """
        Test: conv(R(x)) = R(conv(x)) for rotors R
        
        Uses: CliffordEngine.create_rotor, apply_rotor
        """
        # Stub for testing:
        # x_rotated = clifford_engine.apply_rotor(x, rotor)
        # conv_then_rotate = clifford_engine.apply_rotor(self.forward(x), rotor)
        # rotate_then_conv = self.forward(x_rotated)
        # assert torch.allclose(conv_then_rotate, rotate_then_conv)
```

**Success Criteria**:
- ✓ Heyting mode: Respects orthogonality constraints
- ✓ Boolean mode: No operation restrictions
- ✓ Equivariance: ||conv(R(x)) - R(conv(x))|| < ε
- ✓ Gradients flow through all logic branches

**Logic Engine Usage**:
- **Heyting (1D)**: `check_orthogonality` before every operation
- **Boolean (2D+)**: `verify_excluded_middle` to ensure axioms hold
- **All dims**: `negate` for learning complementary features

---

### Task 20: Complete GNN Architecture

**File**: `graph/models.py` (new)

**Purpose**: Full model using all engines throughout forward pass

**Engine Integration**:
```python
class TripartiteGNN(nn.Module):
    """
    Complete GNN utilizing Logic, Clifford, and Graph engines.
    
    Architecture:
    1. Graph structure encoding (Graph Engine)
    2. Logic constraint filtering (Logic Engine)  
    3. Geometric transformation (Clifford Engine)
    4. Neural processing (Graph Engine)
    
    Every forward pass uses all three engines explicitly.
    """
    
    def __init__(self, dimension, hidden_dim=64):
        self.dimension = dimension
        
        # Initialize all three engines
        self.logic_engine = LogicEngine(dimension)
        self.clifford_engine = CliffordEngine(dimension)
        self.graph_bridge = get_graph_bridge()
        
        # Neural components
        self.geometric_mp = GeometricMessagePassing(...)
        self.logic_conv = LogicAwareConv(dimension)
        
    def forward(self, state: UnifiedState) -> UnifiedState:
        """
        Forward pass through all three algebraic systems.
        
        Flow:
        U → L (check constraints) → C (transform) → G (learn) → U
        """
        # Stub:
        # 1. Logic: Check what operations are valid
        # logic_view = state.as_logic()
        # constraints = logic_engine.get_valid_operations(logic_view)
        
        # 2. Clifford: Apply geometric operations
        # clifford_view = state.as_clifford()
        # transformed = clifford_engine.geometric_product(clifford_view, ...)
        
        # 3. Graph: Learn patterns via message passing
        # graph_view = state.as_graph()
        # learned = self.geometric_mp(graph_view)
        
        # 4. Combine: Use logic to combine results
        # final = logic_engine.join(learned, transformed)
        # return final
```

**Multi-Engine Forward Pass Example**:
```python
def forward_with_explicit_engine_use(self, state):
    """
    Example showing explicit three-engine integration.
    """
    # === LOGIC ENGINE ===
    # Determine operation validity
    logic_state = state.as_logic()
    
    if self.dimension == 1:  # Heyting
        # Check: Can we apply meet to this pair?
        can_meet = self.logic_engine.check_orthogonality(state, other_state)
        if not can_meet:
            # Use join instead (always defined)
            operation_mode = 'join'
    else:  # Boolean
        # All operations valid
        operation_mode = 'meet'
    
    # === CLIFFORD ENGINE ===
    # Apply geometric transformation
    clifford_state = state.as_clifford()
    
    if operation_mode == 'meet':
        # Geometric: wedge product
        result = self.clifford_engine.wedge_product(clifford_state, other_state)
    else:
        # Geometric: inner product
        result = self.clifford_engine.inner_product(clifford_state, other_state)
    
    # === GRAPH ENGINE ===
    # Learn from structure
    graph_state = self.graph_bridge.state_to_graph(result, self.dimension)
    learned = self.geometric_mp(graph_state)
    
    # Convert back through all engines for consistency
    final_state = self.graph_bridge.graph_to_state(learned, self.dimension)
    
    return final_state
```

**Success Criteria**:
- ✓ Every forward pass touches all 3 engines
- ✓ Logic constrains Clifford operations
- ✓ Clifford provides geometric structure to Graph
- ✓ Graph learns patterns respecting Logic+Clifford
- ✓ Training converges (<1000 epochs)

---

## Phase 5B: Logic-First Learning Tasks

**Goal**: Ensure Logic Engine is trained ON, not just routed THROUGH

---

### Task 21A: Logic Learning Tasks

**File**: `training/logic_tasks.py` (new)

**Purpose**: Generate datasets that explicitly teach logical rules

**Three Task Types**:

#### 1. Orthogonality Constraint Learning (Heyting)
```python
def generate_orthogonality_task(n_samples):
    """
    Task: Learn WHEN meet (∧) is defined.
    
    Logic Engine Role: Provide ground truth for operation validity
    Clifford Engine Role: Compute inner product for orthogonality
    Graph Engine Role: Learn to predict validity from structure
    
    Dataset:
    - Input: Pairs of 1D vectors (a, b)
    - Label: True if logic_engine.meet(a,b) is not None
    - Learning goal: Predict orthogonality from graph structure
    """
    samples = []
    logic_engine = LogicEngine(dimension=1)
    clifford_engine = CliffordEngine(dimension=1)
    
    for _ in range(n_samples):
        a = random_1d_vector()
        b = random_1d_vector()
        
        # LOGIC: Is meet defined?
        meet_result = logic_engine.meet(a, b)
        is_valid = (meet_result is not None)
        
        # CLIFFORD: What's the actual inner product?
        inner = clifford_engine.inner_product(a, b)
        
        # GRAPH: Convert to graph for learning
        graph_a = graph_bridge.state_to_graph(a)
        graph_b = graph_bridge.state_to_graph(b)
        
        samples.append({
            'input': (graph_a, graph_b),
            'label': is_valid,
            'explanation': f"Inner product: {inner[0]:.4f}, Valid: {is_valid}"
        })
    
    return samples
```

#### 2. Excluded Middle Learning (Boolean)
```python
def generate_excluded_middle_task(n_samples):
    """
    Task: Learn that a ∨ ¬a = ⊤ in 2D+, but not in 1D.
    
    Logic Engine Role: Compute a ∨ ¬a and check against ⊤
    Clifford Engine Role: Provide geometric interpretation
    Graph Engine Role: Learn dimensional logic transition
    
    Dataset:
    - Input: States in 1D, 2D, 3D
    - Label: Does excluded middle hold?
    - Learning goal: Predict logic type from dimension
    """
    samples = []
    
    for dimension in [1, 2, 3]:
        logic_engine = LogicEngine(dimension)
        
        for _ in range(n_samples // 3):
            state = random_state(dimension)
            
            # LOGIC: Test excluded middle
            holds, deviation = logic_engine.verify_excluded_middle(state)
            
            # CLIFFORD: Show geometric interpretation
            neg_state = logic_engine.negate(state)
            rotation_angle = compute_rotation(state, neg_state)
            
            samples.append({
                'input': state.as_graph(),
                'dimension': dimension,
                'label': holds,
                'rotation': rotation_angle,  # 720° in 1D, 360° in 2D+
                'expected': dimension >= 2
            })
    
    return samples
```

#### 3. Double Negation Learning
```python
def generate_double_negation_task(n_samples):
    """
    Task: Learn ¬¬a ≠ a in 1D, ¬¬a = a in 2D+.
    
    Logic Engine Role: Compute negation with dimension-dependent rules
    Clifford Engine Role: Show as 720°/360° rotation
    Graph Engine Role: Learn negation as graph transformation
    """
    # Similar structure to above
    # Key: Show negation as BOTH logical operation AND geometric rotation
```

**Usage in Training**:
```python
# Training loop explicitly uses logic tasks
def train_with_logic_awareness():
    # 1. Generate logic-specific tasks
    orthog_task = generate_orthogonality_task(1000)
    excluded_task = generate_excluded_middle_task(1000)
    negation_task = generate_double_negation_task(1000)
    
    # 2. Train model to predict logic properties
    for task in [orthog_task, excluded_task, negation_task]:
        train_model_on_logic_task(model, task)
    
    # 3. Verify model learned logical rules
    verify_logic_axioms_hold(model)
```

---

### Task 21B: Multi-Engine Loss Functions

**File**: `training/losses.py` (new)

**Purpose**: Loss function that enforces constraints from all three engines

```python
class TripartiteLoss(nn.Module):
    """
    Loss combining Logic, Clifford, and Graph constraints.
    
    Terms:
    1. Logic consistency: Operations obey dimension-dependent rules
    2. Clifford preservation: Geometric axioms hold
    3. Graph structure: Predictions match graph patterns
    """
    
    def __init__(self, dimension):
        self.logic_engine = LogicEngine(dimension)
        self.clifford_engine = CliffordEngine(dimension)
        
    def forward(self, predicted, target, context):
        """
        Compute tripartite loss.
        
        Each engine contributes a loss term.
        """
        # === LOGIC LOSS ===
        L_logic = self.compute_logic_loss(predicted, context)
        # Penalties:
        # - If 1D and predicted meet on non-orthogonal vectors
        # - If excluded middle violated in 2D+
        # - If double negation wrong for dimension
        
        # === CLIFFORD LOSS ===
        L_clifford = self.compute_clifford_loss(predicted, context)
        # Penalties:
        # - Associativity violation: (ab)c ≠ a(bc)
        # - Grade errors: grade(a∧b) ≠ grade(a) + grade(b)
        # - Norm non-preservation: ||R(a)|| ≠ ||a|| for rotors
        
        # === GRAPH LOSS ===
        L_graph = self.compute_graph_loss(predicted, target)
        # Standard: MSE on node features
        
        # === COMBINED ===
        total = (1.0 * L_graph +      # Primary: fit data
                 0.1 * L_clifford +    # Secondary: preserve geometry
                 0.05 * L_logic)       # Tertiary: respect constraints
        
        return total
```

**Logic Loss Details**:
```python
def compute_logic_loss(self, predicted, context):
    """
    Penalize violation of logical constraints.
    """
    loss = 0.0
    dimension = context['dimension']
    
    # Constraint 1: Heyting (1D) meet constraint
    if dimension == 1:
        # If prediction includes a meet operation
        if context.get('operation') == 'meet':
            # Check if inputs were orthogonal
            a, b = context['operands']
            is_orthogonal = self.logic_engine.check_orthogonality(a, b)
            
            if not is_orthogonal:
                # Penalize: meet should not have been computed
                loss += 10.0  # High penalty
    
    # Constraint 2: Boolean (2D+) excluded middle
    if dimension >= 2:
        # Check: a ∨ ¬a should equal ⊤
        state = context.get('state')
        holds, deviation = self.logic_engine.verify_excluded_middle(state)
        
        if not holds:
            loss += 5.0 * deviation
    
    # Constraint 3: Double negation per dimension
    state = context.get('state')
    not_not_state = self.logic_engine.negate(
        self.logic_engine.negate(state)
    )
    
    if dimension == 1:
        # Should NOT equal original
        if torch.allclose(not_not_state.primary_data, state.primary_data):
            loss += 5.0  # Violation
    else:
        # Should equal original
        if not torch.allclose(not_not_state.primary_data, state.primary_data):
            loss += 5.0
    
    return loss
```

---

## Phase 6: Integrated Reasoning Training

**Goal**: Training loops using all three reasoning modes (I, D, A) across all engines

---

### Task 22: Integrated Training Loop

**File**: `training/integrated_trainer.py` (new)

**Purpose**: Complete training cycle using all (Engine, Mode) combinations

**Architecture**:
```python
class IntegratedReasoningTrainer:
    """
    Trains using all 12 computational modalities.
    
    Each epoch:
    1. INDUCTION phase: Learn from data
       - (G, I): Train GNN on examples
       - (L, I): Learn logical rules from examples
       - (C, I): Discover geometric relationships
       
    2. DEDUCTION phase: Apply learned rules
       - (G, D): Use trained model for predictions
       - (L, D): Apply logical rules to infer
       - (C, D): Use geometric operations to compute
       
    3. ABDUCTION phase: Generate explanations
       - (G, A): Find graph structures that explain errors
       - (L, A): Propose logical rules for failures
       - (C, A): Generate geometric configurations
       
    4. INTEGRATION: Add generated samples back to training
    """
    
    def train_epoch(self, model, data):
        """
        Single epoch using tripartite reasoning.
        """
        # === PHASE 1: INDUCTION ===
        print("INDUCTION: Learning from examples")
        
        # (G, I): Train GNN
        graph_samples = [d.as_graph() for d in data]
        graph_model = self.induction_engine.apply(
            graph_samples,
            Representation.GRAPH,
            context={'epochs': 10}
        )
        
        # (L, I): Learn logic rules
        logic_samples = data  # Already UnifiedStates
        logic_rules = self.learn_logic_rules(logic_samples)
        # Extract: "meet only valid if inner product near 0"
        
        # (C, I): Discover geometry
        clifford_patterns = self.discover_geometric_patterns(data)
        # Extract: "wedge increases grade by 1"
        
        # === PHASE 2: DEDUCTION ===
        print("DEDUCTION: Applying learned rules")
        
        test_data = get_validation_set()
        errors = []
        
        for sample in test_data:
            # (G, D): Predict using trained model
            graph_pred = graph_model(sample.as_graph())
            
            # (L, D): Check logical validity
            logic_pred = self.apply_logic_rules(sample, logic_rules)
            
            # (C, D): Compute geometric result
            clifford_pred = self.compute_geometric(sample, clifford_patterns)
            
            # Compare all three
            if not predictions_agree(graph_pred, logic_pred, clifford_pred):
                errors.append({
                    'sample': sample,
                    'graph': graph_pred,
                    'logic': logic_pred,
                    'clifford': clifford_pred
                })
        
        # === PHASE 3: ABDUCTION ===
        print(f"ABDUCTION: Explaining {len(errors)} errors")
        
        generated_samples = []
        
        for error in errors:
            # (L, A): What logical rule would explain this?
            hypothesis = self.abduce_logic_rule(error)
            # Example: "Maybe meet is invalid here?"
            
            # (C, A): What geometry would produce this?
            geometric_config = self.abduce_geometry(error)
            # Example: "Vectors at 45° angle"
            
            # (G, A): What graph structure matches?
            graph_structure = self.abduce_graph_pattern(error)
            
            # Generate new training sample from hypothesis
            new_sample = self.generate_from_hypothesis(
                hypothesis, geometric_config, graph_structure
            )
            generated_samples.append(new_sample)
        
        # === PHASE 4: INTEGRATION ===
        print(f"INTEGRATION: Adding {len(generated_samples)} samples")
        
        augmented_data = data + generated_samples
        return augmented_data  # Use for next epoch
```

**Key Innovation**: Each phase explicitly uses all three engines

**Example - Learning Orthogonality**:
```python
def learn_orthogonality_integrated():
    """
    Show explicit three-engine, three-mode usage.
    """
    # === EPOCH 1 ===
    
    # INDUCTION: Learn from examples
    # (G, I): Train GNN to predict orthogonality
    graph_model = train_gnn_classifier(pairs_as_graphs, labels)
    
    # (L, I): Learn "meet defined ⟺ orthogonal"
    logic_rule = learn_constraint(pairs, meet_results)
    
    # (C, I): Learn "orthogonal ⟺ inner ≈ 0"
    geometric_threshold = find_threshold(pairs, inner_products)
    
    # DEDUCTION: Test learned knowledge
    # (G, D): Model predicts new pair is orthogonal
    # (L, D): Check if meet is defined
    # (C, D): Compute actual inner product
    # → Compare: Do all three agree?
    
    # ABDUCTION: Generate hard cases
    # Find pairs where predictions disagree
    # (L, A): Generate pair where meet fails unexpectedly
    # (C, A): Generate nearly-orthogonal pair (inner ≈ 0.01)
    # (G, A): Generate graph that confuses model
    
    # INTEGRATION: Retrain on augmented dataset
```

---

### Task 23: Multi-Task Learning with Engine Routing

**File**: `training/multitask_trainer.py` (new)

**Purpose**: Learn multiple tasks, routing to optimal engine per task

**Task Definitions**:
```python
class MultiTaskSpecification:
    """
    Define tasks with engine preferences.
    """
    
    tasks = {
        'orthogonality_check': {
            'preferred_path': [(Logic, Deduction), (Clifford, Deduction)],
            'reason': 'Logic checks constraint, Clifford computes inner',
            'data_need': 'low'  # Can deduce from rules
        },
        
        'wedge_learning': {
            'preferred_path': [(Graph, Induction), (Clifford, Deduction)],
            'reason': 'Learn pattern in Graph, verify in Clifford',
            'data_need': 'high'  # Need examples
        },
        
        'dimension_classification': {
            'preferred_path': [(Logic, Deduction), (Graph, Induction)],
            'reason': 'Logic type determines dimension, Graph learns patterns',
            'data_need': 'medium'
        },
        
        'constraint_discovery': {
            'preferred_path': [(Logic, Induction), (Logic, Abduction)],
            'reason': 'Learn when operations valid, generate edge cases',
            'data_need': 'medium'
        }
    }
```

**Training Strategy**:
```python
def train_multitask(model, task_specs):
    """
    Train on multiple tasks using optimal engine routing.
    """
    for task_name, spec in task_specs.items():
        print(f"\n=== TASK: {task_name} ===")
        
        # Generate task-specific data
        data = generate_task_data(task_name, spec['data_need'])
        
        # Execute preferred engine path
        for (engine, mode) in spec['preferred_path']:
            print(f"  Using: ({engine.value}, {mode.value})")
            
            if engine == Representation.LOGIC:
                result = apply_logic_engine(data, mode)
            elif engine == Representation.CLIFFORD:
                result = apply_clifford_engine(data, mode)
            elif engine == Representation.GRAPH:
                result = apply_graph_engine(data, mode)
            
            # Update model with result
            model.integrate_result(task_name, result)
    
    # Cross-task learning: Share representations
    model.align_task_embeddings()
```

---

## Phase 7: Meta-Learning & Path Optimization

**Goal**: Learn which (Engine, Mode) paths work best for which tasks

---

### Task 24: Path Optimizer with Engine Preferences

**File**: `training/path_optimizer.py` (new)

**Purpose**: Learn task → optimal engine path mapping

**Core Idea**: Different tasks favor different engines

```python
class PathOptimizer:
    """
    Meta-learn optimal paths through combinatorial space.
    
    Tracks:
    - Which engines work best for which task types
    - Which reasoning modes are most effective when
    - Sequential patterns (L→C→G better than G→C→L?)
    """
    
    def __init__(self):
        self.engine_performance = {
            'orthogonality': {
                Logic: 0.95,      # High: Can check directly
                Clifford: 0.90,   # High: Inner product
                Graph: 0.70       # Medium: Must learn
            },
            'wedge_product': {
                Logic: 0.60,      # Low: No direct operation
                Clifford: 0.98,   # Very high: Direct operation
                Graph: 0.85       # High: Can learn pattern
            },
            'dimension_detect': {
                Logic: 0.92,      # High: Logic type = dimension
                Clifford: 0.80,   # Medium: Infer from grade
                Graph: 0.75       # Medium: Learn from structure
            }
        }
        
    def recommend_path(self, task_description):
        """
        Given task, recommend optimal engine sequence.
        
        Returns: [(Engine, Mode), ...]
        """
        task_type = task_description['type']
        data_amount = task_description['data_amount']
        
        # Get engine preferences for this task
        prefs = self.engine_performance.get(task_type, {})
        
        # Build path based on preferences and data
        if data_amount == 'none':
            # Pure deduction with best engine
            best_engine = max(prefs, key=prefs.get)
            return [(best_engine, ReasoningMode.DEDUCTION)]
            
        elif data_amount == 'limited':
            # Learn what we can, deduce the rest
            sorted_engines = sorted(prefs.items(), key=lambda x: x[1], reverse=True)
            return [
                (sorted_engines[0][0], ReasoningMode.DEDUCTION),  # Best: deduce
                (sorted_engines[1][0], ReasoningMode.INDUCTION),  # Second: learn
            ]
            
        else:  # abundant data
            # Learn comprehensively
            return [
                (Representation.GRAPH, ReasoningMode.INDUCTION),   # Learn patterns
                (Representation.CLIFFORD, ReasoningMode.DEDUCTION), # Verify geometry
                (Representation.LOGIC, ReasoningMode.ABDUCTION),   # Generate edge cases
                (Representation.GRAPH, ReasoningMode.INDUCTION),   # Re-learn
            ]
```

**Usage Example**:
```python
# Task: Check if vectors orthogonal, have 5 examples
task = {
    'type': 'orthogonality',
    'data_amount': 'limited',
    'interpretability': 'required'
}

optimizer = PathOptimizer()
path = optimizer.recommend_path(task)
# Returns: [(Logic, D), (Clifford, D)]
# Reason: Logic can check directly, Clifford verifies

# Execute recommended path
result = execute_path(path, data)
```

---

## Phase 8: Verification & Demonstration

**Goal**: Comprehensive examples showing all three engines in use

---

### Task 26: Complete Example Notebooks

**File**: `examples/` (5 notebooks)

Each notebook must explicitly demonstrate all three engines:

#### Notebook 1: `01_three_engines_intro.ipynb`
```markdown
# Introduction to the Tripartite System

## Section 1: Individual Engine Capabilities

### Logic Engine
- Demo: Check orthogonality in 1D (Heyting constraint)
- Demo: Verify excluded middle in 2D (Boolean property)
- Show: Different behavior per dimension

### Clifford Engine  
- Demo: Compute wedge product e1 ∧ e2 = e12
- Demo: Apply rotor rotation
- Show: Grade preservation

### Graph Engine
- Demo: Convert state to graph
- Demo: Message passing preserves structure
- Show: Node features = blade coefficients

## Section 2: Combined Engine Usage

### Example: Orthogonality Check (All Three Engines)
```python
# Logic: Define constraint
logic_engine = LogicEngine(dimension=1)
constraint = lambda a, b: logic_engine.check_orthogonality(a, b)

# Clifford: Compute actual value
clifford_engine = CliffordEngine(dimension=1)
inner = clifford_engine.inner_product(a, b)

# Graph: Learn to predict
graph_model = train_gnn(
    examples_as_graphs,
    labels_from_logic_engine
)

# Verify all three agree
logic_says_orthogonal = constraint(a, b)
clifford_inner_near_zero = (abs(inner[0]) < 0.1)
graph_predicts_orthogonal = graph_model(a.as_graph()) > 0.5

assert logic_says_orthogonal == clifford_inner_near_zero == graph_predicts_orthogonal
```

#### Notebook 2: `02_reasoning_modes_explicit.ipynb`
```markdown
# The Three Reasoning Modes Across All Engines

## Mode 1: Induction (Learn from Examples)

### Logic Induction
Task: Learn when meet is defined (1D)
- Input: [(a₁, b₁, True), (a₂, b₂, False), ...]
- Learn: meet(a, b) defined ⟺ orthogonal(a, b)
- Engine: LogSystemicEngine.check_orthogonality

### Clifford Induction  
Task: Learn wedge product operation
- Input: [(a₁, b₁, c₁), (a₂, b₂, c₂), ...]
- Learn: f(a, b) ≈ a ∧ b
- Engine: CliffordEngine.wedge_product (for labels)

### Graph Induction
Task: Learn geometric patterns
- Input: Graph pairs with labels
- Learn: GNN that predicts operations
- Engine: GraphEngine + message passing

## Mode 2: Deduction (Apply Rules)

### Logic Deduction
Task: Given rules, infer answer
- Input: State a, rule "¬¬a = a in 2D"
- Output: Predicted ¬¬a
- Engine: LogicEngine.negate (twice)

[Continue for Clifford and Graph...]

## Mode 3: Abduction (Generate Explanations)

[Show all three engines generating hypotheses...]
```

#### Notebook 3: `03_combinatorial_paths.ipynb`
```markdown
# Exploring the Combinatorial Space

## Path 1: Logic-First Learning
```python
path = CombinatorialPath([
    (Logic, Induction),    # Learn constraints
    (Clifford, Deduction), # Apply geometry
    (Graph, Induction)     # Learn patterns
])
```
Show: When does this work best?
- Low data scenarios
- Interpretability required
- Known constraints

## Path 2: Graph-First Learning
[Similar for other paths...]

## Path Comparison
Show: Same task, different paths, compare results
```

---

## Summary: Ensuring All Engines Are Utilized

### ✅ Every Task Must Show:

1. **Logic Engine Usage**
   - ❌ NOT JUST: Routing to logic representation
   - ✅ YES: Training on logical properties (orthogonality, excluded middle)
   - ✅ YES: Using constraints to guide learning
   - ✅ YES: Demonstrating Heyting vs Boolean explicitly

2. **Clifford Engine Usage**
   - ❌ NOT JUST: Computing products in isolation
   - ✅ YES: Providing geometric ground truth for learning
   - ✅ YES: Verifying learned models preserve axioms
   - ✅ YES: Equivariance testing with rotors

3. **Graph Engine Usage**
   - ❌ NOT JUST: Converting states to graphs
   - ✅ YES: Learning patterns from graph structure
   - ✅ YES: Message passing that respects geometry and logic
   - ✅ YES: Generalizing to unseen graph configurations

### ✅ Every Reasoning Mode Must Show:

1. **Induction (Learning)**
   - Logic: Learn logical rules from examples
   - Clifford: Learn geometric relationships
   - Graph: Learn structural patterns

2. **Deduction (Application)**  
   - Logic: Apply logical rules to infer
   - Clifford: Apply geometric operations to compute
   - Graph: Apply learned model to predict

3. **Abduction (Generation)**
   - Logic: Generate logical hypotheses for failures
   - Clifford: Generate geometric configurations
   - Graph: Generate graph structures

---

## Implementation Priority

### Critical Path (Do First):
1. **Task 21A**: Logic learning tasks (ensure Logic is trained ON)
2. **Task 18**: Geometric message passing (Graph ← Clifford structure)
3. **Task 21B**: Tripartite loss (all engines constrain learning)
4. **Task 22**: Integrated trainer (all modes in cycle)

### Secondary:
5. **Task 26**: Example notebooks (demonstrate usage)
6. **Task 24**: Path optimizer (meta-learning)

### Polish:
7. **Task 19**: Equivariant layers (nice-to-have)
8. **Task 23**: Multi-task learning (scaling up)

This ensures **Logic is not just present, but essential** to the learning process.
