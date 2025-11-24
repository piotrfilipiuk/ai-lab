from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

# --- 1. The Lattice Definitions ---

@dataclass
class Axis:
    """Represents a mesh axis (e.g., 'data', 'model')."""
    name: str

@dataclass
class ShardingSpec:
    """
    The Lattice State.
    maps: {tensor_dimension: mesh_axis}
    """
    dim_map: Dict[int, str] = field(default_factory=dict)

    def is_sharded(self, dim: int) -> bool:
        return dim in self.dim_map

    def get_axis(self, dim: int) -> Optional[str]:
        return self.dim_map.get(dim)

    def __repr__(self):
        if not self.dim_map:
            return "[Replicated/Unknown]"
        return f"Tiled({self.dim_map})"

    def merge(self, other: 'ShardingSpec') -> 'ShardingSpec':
        """
        The Lattice Join (Merge) operation.
        If conflict, we naively keep the current (or implement priority).
        """
        new_map = self.dim_map.copy()
        changed = False
        for dim, axis in other.dim_map.items():
            if dim not in new_map:
                new_map[dim] = axis
                changed = True
            elif new_map[dim] != axis:
                # TODO: This should raise a Conflict/Reshard requirement.
                # Here, we just log it.
                print(f"  [Conflict] Dim {dim}: {new_map[dim]} vs {axis}")
        return ShardingSpec(new_map), changed

# --- 2. The Graph IR ---

@dataclass
class Tensor:
    name: str
    shape: Tuple[int, ...]
    sharding: ShardingSpec = field(default_factory=ShardingSpec)
    producer: Optional['Op'] = None

class Op:
    def __init__(self, name: str, inputs: List[Tensor], output: Tensor):
        self.name = name
        self.inputs = inputs
        self.output = output
        self.output.producer = self
    
    def propagate(self) -> bool:
        """Returns True if sharding changed."""
        raise NotImplementedError

# --- 3. Transfer Functions (The Propagation Logic) ---

class MatMulOp(Op):
    """
    C = A @ B
    A: [m, k], B: [k, n] -> C: [m, n]
    """
    def propagate(self) -> bool:
        changed = False
        A, B = self.inputs[0], self.inputs[1]
        C = self.output
        
        # --- Forward Propagation (Inputs -> Output) ---
        # 1. Propagate Batch Dim (A[0] -> C[0])
        if A.sharding.is_sharded(0):
            spec = ShardingSpec({0: A.sharding.get_axis(0)})
            new_sharding, chg = C.sharding.merge(spec)
            if chg: 
                C.sharding = new_sharding
                changed = True
                
        # 2. Propagate Feature Dim (B[1] -> C[1])
        if B.sharding.is_sharded(1):
            spec = ShardingSpec({1: B.sharding.get_axis(1)})
            new_sharding, chg = C.sharding.merge(spec)
            if chg:
                C.sharding = new_sharding
                changed = True

        # --- Backward Propagation (Output -> Inputs) ---
        # 1. C[0] implies sharding on A[0]
        if C.sharding.is_sharded(0):
            spec = ShardingSpec({0: C.sharding.get_axis(0)})
            new_sharding, chg = A.sharding.merge(spec)
            if chg:
                A.sharding = new_sharding
                changed = True

        # 2. C[1] implies sharding on B[1]
        if C.sharding.is_sharded(1):
            spec = ShardingSpec({1: C.sharding.get_axis(1)})
            new_sharding, chg = B.sharding.merge(spec)
            if chg:
                B.sharding = new_sharding
                changed = True
                
        return changed

class ElementwiseOp(Op):
    """
    Strict alignment: Input Sharding == Output Sharding
    Used for BiasAdd, ReLU, etc.
    """
    def propagate(self) -> bool:
        changed = False
        tensors = self.inputs + [self.output]
        
        # Collect all known shardings
        combined_spec = ShardingSpec()
        for t in tensors:
            combined_spec, _ = combined_spec.merge(t.sharding)
            
        # Broadcast to all connected tensors
        for t in tensors:
            new_sharding, chg = t.sharding.merge(combined_spec)
            if chg:
                t.sharding = new_sharding
                changed = True
        
        return changed

# --- 4. The Solver (Worklist Algorithm) ---

class ShardingSolver:
    def __init__(self, ops: List[Op]):
        self.ops = ops
    
    def solve(self):
        # Initialize worklist with all ops
        worklist = set(self.ops)
        
        iteration = 0
        while worklist:
            iteration += 1
            op = worklist.pop()
            
            # Run transfer function
            changed = op.propagate()
            
            if changed:
                print(f"Iteration {iteration}: Updated {op.name}")
                # TODO: In a real graph, we would only add neighbors. 
                # For this simple list, we re-add everyone to be safe.
                for neighbor in self.ops:
                    worklist.add(neighbor)

# --- 5. Build and Run: Feed Forward Layer ---

def run_demo():
    # 1. Define Tensors
    # Shapes: Batch=128, Hidden=512, Out=1024
    X = Tensor("X", (128, 512))         # Input
    W = Tensor("W", (512, 1024))        # Weights
    MatMulOut = Tensor("MM_Out", (128, 1024))
    
    b = Tensor("b", (1024,))            # Bias
    AddOut = Tensor("Add_Out", (128, 1024))
    
    Y = Tensor("Y", (128, 1024))        # ReLU Output

    # 2. Define Ops
    op1 = MatMulOp("MatMul", [X, W], MatMulOut)
    op2 = ElementwiseOp("BiasAdd", [MatMulOut, b], AddOut) # Note: b broadcasts, but we treat simple align here
    op3 = ElementwiseOp("ReLU", [AddOut], Y)

    # 3. Scenario: User annotates Data Parallelism on Input X
    # "Split the Batch dimension (0) across 'mesh_x'"
    print("--- Initial State ---")
    X.sharding = ShardingSpec({0: "mesh_x"})
    
    all_tensors = [X, W, MatMulOut, b, AddOut, Y]
    for t in all_tensors: print(f"{t.name}: {t.sharding}")

    # 4. Run Solver
    print("\n--- Running Propagation ---")
    solver = ShardingSolver([op1, op2, op3])
    solver.solve()

    # 5. Result
    print("\n--- Final State ---")
    for t in all_tensors: print(f"{t.name}: {t.sharding}")

if __name__ == "__main__":
    run_demo()
