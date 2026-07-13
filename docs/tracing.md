# Tracing the Compilation Process

The MQT Predictor framework allows you to extract detailed, step-by-step
information about the reinforcement learning agent's compilation process by
generating a comprehensive JSON trace.

## Enabling the Tracer

By default, the compilation environment acts as a black box: it takes an
uncompiled circuit and returns the final compiled version alongside a list of
the applied passes. To minimize overhead during standard execution, detailed
step-by-step logging is _disabled_.

To illuminate this process and track the exact sequence of transformations and
intermediate circuit metrics, you can enable the compilation tracer. This is
done by simply passing a valid file path to the `tracer_output_path` argument
when calling the `qcompile` function.

Here is an example of how to compile a 5-qubit GHZ state while simultaneously
generating a trace file:

```python
from mqt.predictor import qcompile
from mqt.bench import get_benchmark, BenchmarkLevel

# 1. Obtain an uncompiled quantum circuit
uncompiled_qc = get_benchmark("ghz", level=BenchmarkLevel.ALG, circuit_size=5)

# 2. Compile the circuit and enable tracing
compiled_qc, compilation_info, selected_device = qcompile(
    uncompiled_qc, figure_of_merit="expected_fidelity", tracer_output_path="./compilation_trace.json"
)
```

## Trace File Structure

The generated trace file is a standard JSON document that closely mirrors the
internal Python dataclasses used to track the compilation. The file is
structured into three main components: top-level compilation metadata, target
device information, and a sequential list of compilation steps.

### Top-Level Metadata

The root of the JSON file provides general context about the compilation run:

- **`circuit_name`**: The name of the original uncompiled circuit.
- **`figure_of_merit`**: The primary metric the RL agent optimized for (e.g.,
  `expected_fidelity`).
- **`mdp_policy`**: The Markov Decision Process (MDP) transition policy used
  during the episode.
- **`schema_version`**: The version of the JSON schema (useful for ensuring
  compatibility with visualization tools like MQT FlowViz).
- **`timestamp`**: The exact Unix timestamp when the compilation started.
- **`total_duration`**: The total time (in seconds) taken to execute the entire
  compilation pipeline.

### Target Device Information (`device`)

The `device` object captures a snapshot of the backend hardware the circuit is
being compiled for. This ensures the trace is entirely self-contained, even if
the actual hardware specifications change in the future.

- **`name`**: The identifier of the target device (e.g., `ibm_washington`).
- **`device_qubits`**: The total number of physical qubits available on the
  device.
- **`native_gates`**: A list of the basic gate instructions supported natively
  by the hardware.
- **`topology`**: A list of directional edges (defined by `control` and `target`
  qubit indices) representing the device's connectivity graph.
- **`calibration_data`**: Detailed error rates and execution durations for every
  native gate applied to specific qubit combinations.

### Compilation Steps (`steps`)

The `steps` array is the core of the trace file. It contains an ordered sequence
of snapshots, with each entry representing the state of the circuit and the RL
environment _after_ a specific compilation pass has been applied.

Each step includes:

- **Action Details**:
  - `step_index`: The chronological step number.
  - `action_name`: The specific compiler pass applied (e.g.,
    `OptimizeCliffords`).
  - `action_type`: The category of the pass (e.g., Optimization, Routing,
    Synthesis).
  - `action_duration`: How long the pass took to execute.
  - `reward`: The numeric reward granted to the RL agent for choosing this
    action.
- **Circuit Representation**:
  - `circuit_qasm3`: The physical structure of the quantum circuit at this exact
    timestep, serialized in standard OpenQASM 3.0 format.
- **High-Level Statistics**:
  - Simple counts like `current_depth`, `num_qubits`, and `total_gates`.
  - `gates_per_operation`: A dictionary breaking down the frequency of each
    specific gate type currently present in the circuit.
- **State Flags**: Boolean markers (`synthesized`, `laid_out`, `routed`)
  indicating the circuit's progress through the necessary compilation phases,
  ending with `is_terminal` when the process concludes.
- **Environment Features & Metrics**:
  - **Figures of Merit**: The evaluated metrics for the current state (e.g.,
    Expected Fidelity, Critical Depth, Hellinger Distance, Success Probability).
    Each metric includes its calculated `value` and the `kind` of calculation
    used (exact vs. approximated).
  - **Feature Vector**: The specific circuit characteristics observed by the RL
    agent to make its next decision, including `program_communication`,
    `raw_critical_depth`, `entanglement_ratio`, `parallelism`, and `liveness`.
