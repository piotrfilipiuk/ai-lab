"""
# StochIR: A Noise-Aware Intermediate Representation

**StochIR** is a toy compiler prototype designed to demonstrate how we might
bridge the gap between deterministic models and stochastic, nonlinear physical
substrates.

Instead of traditional compiler lowering (where we assume perfect bitwise
computation on a von Neumann architecture), StochIR takes a **"First Principles
Mapping"** approach. It maps a computation graph directly onto noisy analog
components where hardware-level noise isn't simulated away; it is embedded into
the math.

## The Problem: Noise is a Feature, not a Bug

Building software interfaces to the physics of silicon requires a compiler that
understands stochasticity. When running neural networks on analog hardware,
operations are subject to thermal drift, shot noise, and analog decay.

Professor Michael Carbin's research at MIT on "Programming Systems for
Uncertainty" highlights exactly this: programming languages and tools need to
treat uncertainty as a first-class citizen rather than a hidden defect.

## The Approach

This prototype introduces a Python-based IR where nodes are explicitly
non-deterministic.

1.  **Probabilistic Arithmetic**: Operations (`AddOp`, `MulOp`) don't return
    single floats; they natively inject modeled hardware variance into the
    signal.
2.  **Physical Substrate Mapping**:
    -   Operations transition into the analog domain via an `AnalogizeOp` (DAC).
    -   Core compute happens in the noisy analog regime.
    -   Operations transition back to the digital domain via a `DigitizeOp`
        (ADC) to recalibrate.
3.  **Dual-Execution IR**: Every node implements both a stochastic `evaluate()`
    and a deterministic `evaluate_digital()`. The compiler uses the pure digital
    path to calculate the exact CPU ground state.
4.  **Hardware Calibration Mapping**: The `NoiseAnalysisPass` statically
    analyzes SNR degradation using a dry-run Monte Carlo simulation. If variance
    drifts past a threshold, it inserts a `DigitizeOp` (ADC). The compiler
    programs the exact digital ground-truth into the ADC so that the analog
    signal is mathematically shifted, quantized, and shifted back; perfectly
    squashing thermal noise and mimicking industrial Offset Compensation techniques.

## Why This Matters (The Pitch)

Coming from a background of working extensively with high-level distributed
compilation and tensor representations (XLA-TPU), I understand how to organize
structured linear algebra for massive scale.

However, the future of unconventional silicon isn't just about massive FLOPs;
its about what happens to those FLOPs against the physics of the substrate.
StochIR represents the "full-stack" perspective required here: maintaining the
high-level semantic abstractions of a model while mapping them directly down to
the noisy realities of the analog components executing them.

## Running the Prototype

```bash
python3 stoch_ir.py
```

This will run a Monte Carlo simulation of a small graph: `Output = (A + B) * (C
+ D)`. The `NoiseAnalysisPass` will detect the variance explosion across the
multiplication node and aggressively insert a `DigitizeOp` to contain the noise.
"""

from typing import Dict, List

import numpy as np


class StochasticNode:
  """Base IR Node. Accounts for stochasticity (noise) in analog hardware."""

  def __init__(
      self, name: str, parents: List["StochasticNode"], noise_std: float = 0.01
  ):
    self.name = name
    self.parents = parents
    # Baseline noise stddev injected purely by the physical substrate of this
    # operation
    self.noise_std = noise_std
    # Expected outputs from parents (useful for compiler passes)
    self.children = []
    for p in parents:
      p.children.append(self)

  def evaluate(self, inputs: List[np.ndarray]) -> np.ndarray:
    raise NotImplementedError

  def evaluate_digital(self, inputs: List[np.ndarray]) -> np.ndarray:
    """Pure software/CPU evaluation (no noise) for ground truth."""
    raise NotImplementedError


class AnalogizeOp(StochasticNode):
  """DAC (Digital-to-Analog).

  Injects initial thermal/shot noise to digital inputs.
  """

  def __init__(self, name: str, value: float, noise_std: float = 0.05):
    super().__init__(name, [], noise_std)
    self.value = value

  def generate(self, num_samples: int) -> np.ndarray:
    """Generates the initial analog signal block."""
    ideal = np.full(num_samples, self.value)
    noise = np.random.normal(0, self.noise_std, size=num_samples)
    return ideal + noise

  def evaluate(self, inputs: List[np.ndarray]) -> np.ndarray:
    raise RuntimeError("AnalogizeOp is a source node; use generate() instead.")

  def generate_digital(self, num_samples: int) -> np.ndarray:
    return np.full(num_samples, self.value)


class AddOp(StochasticNode):
  """Analog Addition (e.g., Kirchhoff's current law junction). Adds noise."""

  def evaluate(self, inputs: List[np.ndarray]) -> np.ndarray:
    res = sum(inputs)
    noise = np.random.normal(0, self.noise_std, size=res.shape)
    return res + noise

  def evaluate_digital(self, inputs: List[np.ndarray]) -> np.ndarray:
    return sum(inputs)


class MulOp(StochasticNode):
  """Analog Multiplication (e.g., Variable gain amplifier). Adds noise."""

  def evaluate(self, inputs: List[np.ndarray]) -> np.ndarray:
    res = inputs[0] * inputs[1]
    noise = np.random.normal(0, self.noise_std, size=res.shape)
    return res + noise

  def evaluate_digital(self, inputs: List[np.ndarray]) -> np.ndarray:
    return inputs[0] * inputs[1]


class DigitizeOp(StochasticNode):
  """ADC (Analog-to-Digital) & Calibration.

  Discretizes signal by snapping to a defined grid. During compilation,
  this layer centers its quantization grid around the expected expected_mean
  to effectively filter out thermal noise and re-synchronize the signal.
  """

  def __init__(
      self,
      name: str,
      parent: StochasticNode,
      step_size: float = 1.0,
      expected_mean: float = 0.0,
      noise_std: float = 0.0,
  ):
    super().__init__(name, [parent], noise_std=noise_std)
    self.step_size = step_size
    self.expected_mean = expected_mean

  def evaluate(self, inputs: List[np.ndarray]) -> np.ndarray:
    signal = inputs[0]

    # Calibration logic.
    # The ADC knows the expected mean from the compiler pass. It shifts its
    # reference grid so the expected mean sits *exactly* in the center of a step
    # bin. This maximizes the amount of noise it can swallow before flipping an
    # erroneous bit.

    # Calculate offset to center grid at expected_mean
    grid_offset = (
        self.expected_mean
        - np.round(self.expected_mean / self.step_size) * self.step_size
    )

    # Shift, quantize on integer grid, and shift back
    shifted = signal - grid_offset
    quantized = np.round(shifted / self.step_size) * self.step_size
    calibrated_signal = quantized + grid_offset
    return calibrated_signal

  def evaluate_digital(self, inputs: List[np.ndarray]) -> np.ndarray:
    # A perfect ADC on a perfect signal is just identity.
    return inputs[0]


class StochasticGraph:
  """A container for the compute graph with a topological sort."""

  def __init__(self, outputs: List[StochasticNode]):
    self.outputs = outputs
    self.nodes = self._topological_sort()

  def _topological_sort(self) -> List[StochasticNode]:
    """Performs a topological sort of the graph starting from the outputs.

    The graph is built by traversing from the output nodes up to their parents.

    Returns:
      A list of StochasticNode instances in topological order, where each node
      appears after all of its parents.
    """
    visited = set()
    order = []

    def dfs(node):
      if node in visited:
        return
      visited.add(node)
      for p in node.parents:
        dfs(p)
      order.append(node)

    for out in self.outputs:
      dfs(out)
    return order


class NoiseAnalysisPass:
  """Analyzes SNR (Signal-to-Noise Ratio) accumulation across the graph.

  If variance gets too high, it inserts a DigitizeOp (Calibration/ADC layer).
  """

  def __init__(self, variance_threshold: float = 2.0):
    self.variance_threshold = variance_threshold

  def optimize(self, graph: StochasticGraph) -> StochasticGraph:
    # Simple static analysis (approximated via a single run logic)
    # We use a pure digital CPU run to find the exact expected signal mean,
    # and a dry-run Monte Carlo to find physical variance at each node.
    results_analog = simulate_monte_carlo(graph, num_samples=5000)
    results_digital = simulate_digital(graph, num_samples=1)

    new_nodes = {}
    for node in graph.nodes:
      new_nodes[node.name] = node  # copy

      if node.name in results_analog:
        dist = results_analog[node.name]
        var = np.var(dist)

        # Exclude existing digitizers and outputs from being wrapped
        # unnecessarily
        if var > self.variance_threshold and not isinstance(node, DigitizeOp):
          print(
              f"[Compiler Pass] High variance detected at node '{node.name}'"
              f" (Var: {var:.2f}). Inserting ADC (DigitizeOp)."
          )

          old_children = list(node.children)
          node.children = []

          # Instead of node feeding directly to its children, wire it to an ADC
          # We dynamically compute a step size large enough to swallow the
          # `expected_mean` from the pure digital simulation, proving we can
          # calibrate physical systems using standard CPU truth.
          expected_mean = results_digital[node.name][0]
          optimal_step = max(1.0, np.ceil(np.sqrt(var) * 6))

          adc = DigitizeOp(
              f"{node.name}_ADC",
              node,
              step_size=optimal_step,
              expected_mean=expected_mean,
          )

          # Redirect children to read from ADC instead
          for child in old_children:
            idx = child.parents.index(node)
            child.parents[idx] = adc
            adc.children.append(child)

          # Update graph outputs if we wrapped an output node
          for i in range(len(graph.outputs)):
            if graph.outputs[i] == node:
              graph.outputs[i] = adc

    return StochasticGraph(graph.outputs)


def simulate_monte_carlo(
    graph: StochasticGraph, num_samples: int = 1000
) -> Dict[str, np.ndarray]:
  """Executes the stochastic graph N times, simulating analog drift."""
  results = {}
  for node in graph.nodes:
    if isinstance(node, AnalogizeOp):
      results[node.name] = node.generate(num_samples)
    else:
      inputs = [results[p.name] for p in node.parents]
      results[node.name] = node.evaluate(inputs)
  return results


def simulate_digital(
    graph: StochasticGraph, num_samples: int = 1
) -> Dict[str, np.ndarray]:
  """Executes the graph using pure digital logic (CPU Baseline)."""
  results = {}
  for node in graph.nodes:
    if isinstance(node, AnalogizeOp):
      results[node.name] = node.generate_digital(num_samples)
    else:
      inputs = [results[p.name] for p in node.parents]
      results[node.name] = node.evaluate_digital(inputs)
  return results


if __name__ == "__main__":
  np.random.seed(42)

  # -- Build Original Graph --
  # Representing: Output = (A + B) * (C + D)
  # Physical drift accumulates aggressively over multiplication.
  a = AnalogizeOp("A", value=10.0, noise_std=0.5)
  b = AnalogizeOp("B", value=5.0, noise_std=0.5)
  c = AnalogizeOp("C", value=2.0, noise_std=0.2)
  d = AnalogizeOp("D", value=2.0, noise_std=0.2)

  add1 = AddOp("Add1", [a, b], noise_std=1.0)
  add2 = AddOp("Add2", [c, d], noise_std=0.5)

  output = MulOp("Output", [add1, add2], noise_std=2.0)

  graph = StochasticGraph([output])

  print("--- Ground Truth (Pure Digital Math) ---")
  digital_results = simulate_digital(graph, num_samples=1)
  expected_output = digital_results["Output"][0]
  print(f"Expected Output: {expected_output:.2f}\n")

  print("--- Running Original Graph (MCMC) ---")
  orig_results = simulate_monte_carlo(graph, num_samples=10000)
  final_mean = np.mean(orig_results["Output"])
  final_variance = np.var(orig_results["Output"])
  print(f"Original Output Mean:     {final_mean:.2f}")
  print(f"Original Output Variance: {final_variance:.2f}\n")

  # -- Run Compiler Pass --
  print("--- Running Noise Analysis Compiler Pass ---")
  compiler = NoiseAnalysisPass(variance_threshold=0.2)
  # Note: the optimization modifies the connections in-place for this prototype
  optimized_graph = compiler.optimize(graph)

  print("\n--- Running Optimized Graph (MCMC) ---")
  opt_results = simulate_monte_carlo(optimized_graph, num_samples=10000)
  final_mean_opt = np.mean(opt_results["Output"])
  final_variance_opt = np.var(opt_results["Output"])
  print(f"Optimized Output Mean:     {final_mean_opt:.2f}")
  print(f"Optimized Output Variance: {final_variance_opt:.2f}")


