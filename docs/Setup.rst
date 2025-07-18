MQT Predictor Framework Setup
=============================

To use the MQT Predictor framework, you must first configure the devices and train the reinforcement learning and supervised machine learning models. This section outlines how to do that.

Step 1: Add Devices
-------------------

All devices supported by `MQT Bench <https://github.com/cda-tum/mqt-bench>`_ are natively supported. You can retrieve and inspect available devices as follows:

.. code-block:: python

    from mqt.bench.targets import get_device, get_available_device_names

    for name in get_available_device_names():
        dev = get_device(name)
        print(f"{name}: {dev.num_qubits} qubits")

Custom devices are also supported as long as they are defined as Qiskit ``Target`` objects.

Step 2: Train Reinforcement Learning Models
-------------------------------------------

For each device, a dedicated reinforcement learning (RL) model must be trained. This is based on a figure of merit and a set of training circuits in QASM format located in
`mqt/predictor/rl/training_data/training_circuits <https://github.com/munich-quantum-toolkit/predictor/tree/main/src/mqt/predictor/rl/training_data/training_circuits>`_.

.. code-block:: python

    from mqt.predictor.rl import Predictor as RLPredictor
    from mqt.bench.targets import get_target

    device = get_target("ibm_falcon_27")
    rl_pred = RLPredictor(device=device, figure_of_merit="expected_fidelity")
    rl_pred.train_model(timesteps=100000, model_name="ibm_falcon_27_model")

In addition to ``expected_fidelity``, also ``critical_depth`` is supported. New figures of merit can be added to ``mqt.predictor.reward``.

To register additional compilation passes (e.g., from Qiskit, TKET, or BQSKit), use:

.. code-block:: python

    from mqt.predictor.rl.actions import (
        CompilationOrigin,
        DeviceIndependentAction,
        PassType,
        register_action,
    )

    my_custom_pass = ...  # e.g., a Qiskit pass
    action = DeviceIndependentAction(
        name="custom_action",
        pass_type=PassType.OPT,
        transpile_pass=[my_custom_pass],
        origin=CompilationOrigin.QISKIT,
    )
    register_action(action)

For other compilation sources, a new ``CompilationOrigin`` must be defined and conversion to/from Qiskit's ``QuantumCircuit`` implemented.

Step 3: Generate Training Data and Train ML Model
-------------------------------------------------

Once the RL models are trained, generate the training data and train the supervised ML model using:

.. code-block:: python

    from mqt.predictor.ml import setup_device_predictor
    from mqt.bench.targets import get_device

    devices = [
        get_device("ibm_falcon_27"),
        get_device("ibm_eagle_127"),
        get_device("quantinuum_h2_56"),
    ]
    setup_device_predictor(
        devices=devices,
        figure_of_merit="expected_fidelity",
    )

This function will:

#. Compile all uncompiled training circuits using the RL models.
#. Generate training data from the compiled circuits.
#. Train and save a supervised model for device prediction.

You can optionally specify custom paths for uncompiled and compiled QASM files using the ``path_uncompiled_circuits`` and ``path_compiled_circuits`` arguments.

Step 4: Compile a Circuit with ``qcompile``
-------------------------------------------

After setup, any quantum circuit can be compiled for the most suitable device with:

.. code-block:: python

    from mqt.predictor import qcompile
    from mqt.bench import get_benchmark, BenchmarkLevel

    uncompiled_qc = get_benchmark("ghz", level=BenchmarkLevel.ALG, circuit_size=5)
    compiled_qc, compilation_info, selected_device = qcompile(
        uncompiled_qc, figure_of_merit="expected_fidelity"
    )

This returns:

- the compiled quantum circuit,
- the compilation metadata, and
- the selected device.

``qcompile`` combines automatic device selection with device-specific compilation based on the selected figure of merit.
