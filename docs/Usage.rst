Repository Usage
================
There are two ways how to use MQT Predictor:

#. Via the pip package ``mqt.predictor``
#. Directly via this repository

Usage via pip package
---------------------

MQT Predictor is available via `PyPI <https://pypi.org/project/mqt.predictor/>`_

.. code-block:: console

   (venv) $ pip install mqt.predictor

To compile a quantum circuit, use the ``qcompile`` method:

.. automodule:: mqt.predictor
    :members: qcompile

Currently available figures of merit are ``expected_fidelity`` and ``critical_depth``.

An example how ``qcompile`` is used can be found in the :doc:`quickstart <Quickstart>` jupyter notebook.

.. _pip_usage:

Usage directly via this repository
----------------------------------

For that, the repository must be cloned and installed:

.. code-block::

   git clone https://github.com/munich-quantum-toolkit/predictor.git
   cd mqt-predictor
   pip install .

Afterwards, the package can be used as described :ref:`above <pip_usage>`.

MQT Predictor Framework Setup
=============================
To run ``qcompile``, the MQT Predictor framework must be set up. How this is properly done is described next.

First, the to-be-considered quantum devices must be included in the framework.
All devices supported by `MQT Bench <https://github.com/cda-tum/mqt-bench>`_ are natively supported.
Furthermore, a custom device can be added to the framework as long as it is provided as a Qiskit Target object.

Second, for each device, a respective reinforcement learning model must be trained. This is done by running
the following command based on the training data in the form of quantum circuits provided as qasm files in
`mqt/predictor/rl/training_data/training_circuits <https://github.com/munich-quantum-toolkit/predictor/tree/main/src/mqt/predictor/rl/training_data/training_circuits>`_:

.. code-block:: python

    import mqt.predictor
    from mqt.bench.targets import get_target

    device = get_target(
        "ibm_falcon_27"
    )  # or any other device given as a Qiskit Target object
    rl_pred = mqt.predictor.rl.Predictor(
        device=device,
        figure_of_merit="expected_fidelity",
    )
    rl_pred.train_model(timesteps=100000, model_name="sample_model_rl")

This will train a reinforcement learning model for the ``ibm_falcon_27`` device with the expected fidelity as figure of merit.
Additionally to the expected fidelity, also critical depth is provided as another figure of merit.
Further figures of merit can be added in `mqt.predictor.reward.py <https://github.com/munich-quantum-toolkit/predictor/tree/main/src/mqt/predictor/reward.py>`_.
Please note that there is a pre-configured set of available compilation passes that are supported.
This is defined in `mqt.predictor.rl.actions <https://github.com/munich-quantum-toolkit/predictor/tree/main/src/mqt/predictor/rl/actions.py>`_ and can be easily extended.
If another compilation pass from Qiskit, TKET, or BQSKit shall be added, this can be done using:

.. code-block:: python

    from mqt.predictor.rl.actions import (
        CompilationOrigin,
        DeviceIndependentAction,
        PassType,
        register_action,
    )

    my_custom_pass = ...  # Define your custom pass here, e.g., a Qiskit pass
    action = DeviceIndependentAction(
        name="test_action",
        pass_type=PassType.OPT,
        transpile_pass=[my_custom_pass],
        origin=CompilationOrigin.QISKIT,
    )
    register_action(action)

For other sources, defining a new ``CompilationOrigin`` is necessary as well as providing parsing methods to and from Qiskit's QuantumCircuit, since this is used as our internal representation of quantum circuits.

Third, after the reinforcement learning models that are used for the respective compilations are trained, the
supervised machine learning model to predict the device selection must be trained.
This is done by first creating the necessary training data (based on the training data in the form of quantum circuits provided as qasm files in
`mqt/predictor/ml/training_data/training_circuits <https://github.com/munich-quantum-toolkit/predictor/tree/main/src/mqt/predictor/ml/training_data/training_circuits>`_) and then running the following command:

.. code-block:: python

    device = get_device("ibm_falcon_27")
    ml_pred = mqt.predictor.ml.Predictor(
        devices=[device], figure_of_merit="expected_fidelity"
    )
    ml_pred.generate_compiled_circuits(timeout=600)  # timeout in seconds
    training_data, name_list, scores_list = ml_pred.generate_trainingdata_from_qasm_files()
    ml_pred.save_training_data(
        training_data,
        name_list,
        scores_list,
    )

This will compile all provided uncompiled training circuits for all available devices and figures of merit.
Furthermore, a selection of the devices could be used, such as ``devices=[get_device("ibm_falcon_27"), get_device("ibm_eagle_127"), get_device("quantinuum_h2_56")]``.
Afterwards, the training data is generated individually for a figure of merit.


The devices currently supported by `MQT Bench <https://github.com/cda-tum/mqt-bench>`_ can by accessed via:

.. code-block:: python

    from mqt.bench.targets import get_device, get_available_device_names

    for num, device_name in enumerate(get_available_device_names()):
        print(f"{num+1}: {device_name} with {get_device(device_name).num_qubits} qubits")



This training data can then be saved and used to train the supervised machine learning model:
.. code-block:: python

    ml_pred.train_random_forest_classifier()

Finally, the MQT Predictor framework is fully set up and can be used to predict the most
suitable device for a given quantum circuit using supervised machine learning and compile
the circuit for the predicted device using reinforcement learning by running:

.. code-block:: python

    from mqt.predictor import qcompile
    from mqt.bench import get_benchmark, BenchmarkLevel

    qc_uncompiled = get_benchmark(benchmark="ghz", level=BenchmarkLevel.ALG, circuit_size=5)
    compiled_qc, compilation_information, device = qcompile(
        uncompiled_qc, figure_of_merit="expected_fidelity"
    )


This returns the compiled quantum circuit for the predicted device together with additional information of the compilation procedure.
