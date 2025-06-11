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
Currently, all devices supported by `MQT Bench <https://github.com/cda-tum/mqt-bench>`_ are natively supported.
In case another device shall be considered, it can be added by using a similar format as in MQT Bench but it is not
necessary to add it in the repository since it can be directly added to the MQT Predictor framework as follows:

- Modify in `mqt/predictor/rl/predictorenv.py <https://github.com/munich-quantum-toolkit/predictor/tree/main/src/mqt/predictor/rl/predictorenv.py>`_. the line where ``mqt.bench.devices.get_device_by_name`` is used.
- Modify in `mqt/predictor/ml/predictor.py <https://github.com/munich-quantum-toolkit/predictor/tree/main/src/mqt/predictor/ml/predictor.py>`_. the lines where ``mqt.bench.devices.*`` are used.
- Follow the same data format as defined in `mqt.bench.devices.device.py <https://github.com/cda-tum/mqt-bench/tree/main/src/mqt/bench/devices/device.py>`_

Second, for each supported device, a respective reinforcement learning model must be trained. This is done by running
the following command based on the training data in the form of quantum circuits provided as qasm files in
`mqt/predictor/rl/training_data/training_circuits <https://github.com/munich-quantum-toolkit/predictor/tree/main/src/mqt/predictor/rl/training_data/training_circuits>`_:

.. code-block:: python

    import mqt.predictor

    rl_pred = mqt.predictor.rl.Predictor(
        figure_of_merit="expected_fidelity", device_name="ibm_washington"
    )
    rl_pred.train_model(timesteps=100000, model_name="sample_model_rl")

This will train a reinforcement learning model for the ``ibm_washington`` device with the expected fidelity as figure of merit.
Additionally to the expected fidelity, also critical depth is provided as another figure of merit.
Further figures of merit can be added in `mqt.predictor.reward.py <https://github.com/munich-quantum-toolkit/predictor/tree/main/src/mqt/predictor/reward.py>`_.

Third, after the reinforcement learning models that are used for the respective compilations are trained, the
supervised machine learning model to predict the device selection must be trained.
This is done by first creating the necessary training data (based on the training data in the form of quantum circuits provided as qasm files in
`mqt/predictor/ml/training_data/training_circuits <https://github.com/munich-quantum-toolkit/predictor/tree/main/src/mqt/predictor/ml/training_data/training_circuits>`_) and then running the following command:

.. code-block:: python

    ml_pred = mqt.predictor.ml.Predictor(figure_of_merit="expected_fidelity")
    ml_pred.generate_compiled_circuits(timeout=600)  # timeout in seconds
    training_data, name_list, scores_list = ml_pred.generate_trainingdata_from_qasm_files()
    ml_pred.save_training_data(
        training_data,
        name_list,
        scores_list,
    )

This will compile all provided uncompiled training circuits for all available devices and figures of merit.
However, only devices with a bidirectional connectivity are supported.
To this end, the following the devices currently supported by [MQT Bench](https://github.com/cda-tum/mqt-bench) are supported:

- ibm_washington
- ibm_montreal
- ionq_harmony
- ionq_aria1
- rigetti_aspen_m3
- quantinuum_h2
- iqm_adonis
- iqm_apollo

Furthermore, a selection of the devices could be used, such as ``devices=["ibm_montreal", "ibm_washington", "ionq_aria1"]``.
Afterwards, the training data is generated individually for a figure of merit.
This training data can then be saved and used to train the supervised machine learning model:

.. code-block:: python

    ml_pred.train_random_forest_classifier()

Finally, the MQT Predictor framework is fully set up and can be used to predict the most
suitable device for a given quantum circuit using supervised machine learning and compile
the circuit for the predicted device using reinforcement learning by running:

.. code-block:: python

    from mqt.predictor import qcompile
    from mqt.bench import get_benchmark

    qc_uncompiled = get_benchmark(benchmark_name="ghz", level="alg", circuit_size=5)
    compiled_qc, compilation_information, device = qcompile(
        uncompiled_qc, figure_of_merit="expected_fidelity"
    )


This returns the compiled quantum circuit for the predicted device together with additional information of the compilation procedure.
