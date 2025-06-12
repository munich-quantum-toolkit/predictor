Automatic Device Selection
==========================

To realize any quantum application, a suitable quantum device must be selected for the execution of the developed quantum algorithm.
This alone is non-trivial since new quantum devices based on various underlying technologies emerge on an almost daily basis—each with their own advantages and disadvantages.
There are hardly any practical guidelines on which device to choose based on the targeted application.
As such, the best guess in many cases today is to simply try out many (if not all) possible devices and, afterwards, choose the best results—certainly a time- and resource-consuming endeavor that is not sustainable for the future.

A naive approach to select the best quantum device for a given quantum circuit would be to compile it for all devices, e.g., using the trained RL models which act as specialized compilers for supported quantum devices.
Afterwards, the resulting compiled circuits must be evaluated according to some figure of merit to identify the most promising device.
However, doing this for each and every to-be-compiled quantum circuit is practically infeasible since compilation is a time-consuming task.

The MQT Predictor framework provides an easy-to-use solution to this problem by using supervised machine learning.
It learns from previous compilations of other quantum circuits and models the problem of determining the most promising device for a circuit and figure of merit as a statistical classification task—a task well suited for supervised machine learning.
For that, the framework is trained with based on three inputs:

1. Training circuits
2. The compilation options for all supported devices
3. The figure of merit to optimize for

.. image:: /_static/ml.png
   :width: 100%
   :alt: Illustration of the ML model
   :align: center

The trained model then acts as a predictor and can be used to predict the most suitable device for a given quantum circuit and figure of merit.

.. _supported-quantum-devices:

Supported Quantum Devices
-------------------------
Currently, devices based on two qubit technologies are supported:

- Superconducting-based:
    - IQM Adonis with 5 qubits
    - IQM Apollo with 20 qubits
    - IBM Montreal with 27 qubits
    - Quantinuum H2 with 32 qubits
    - Rigetti Aspen-M3 with 80 qubits
    - IBM Washington with 127 qubits
- Ion Trap-based:
    - IonQ Harmony with 11 qubits
    - IonQ Aria1 with 25
    - Quantinuum H2 with 32 qubits


Adding further devices is straight-forward and requires only to provide its native gate-set, connectivity, and calibration data.

Evaluated Machine Learning Classifiers
--------------------------------------

For the evaluation of our methodology, seven supervised machine learning classifiers have been used:

- Random Forest
- Gradient Boosting
- Decision Tree
- Nearest Neighbor
- Multilayer Perceptron
- Support Vector Machine
- Naive Bayes

In our exemplary scenario, the Random Forest classifier achieved the best performance.
To play around with all the examined models, please use the `Jupyter notebook <https://github.com/munich-quantum-toolkit/predictor/blob/main/evaluations/supervised_ml_models/evaluation.ipynb>`_.

Training Data
-------------

To train the model, sufficient training data must be provided as qasm files in the `respective directory <https://github.com/munich-quantum-toolkit/predictor/tree/main/src/mqt/predictor/ml/training_data/training_circuits>`_.
We provide the training data used in the initial performance evaluation of this framework.

After the adjustment is finished, the following methods need to be called to generate the training data:

.. code-block:: python

    from mqt.predictor import ml

    predictor = ml.Predictor()
    predictor.generate_compiled_circuits(figure_of_merit="expected_fidelity")
    training_data, name_list, scores_list = predictor.generate_trainingdata_from_qasm_files(
        figure_of_merit="expected_fidelity"
    )
    ml.helper.save_training_data(
        training_data, name_list, scores_list, figure_of_merit="expected_fidelity"
    )

Now, the Random Forest classifier can be trained:

.. code-block:: python

    predictor.train_random_forest_model(figure_of_merit="expected_fidelity")


Additionally, the raw training data may be extracted and can be used for any machine learning model:

.. code-block:: python

    training_data = predictor.get_prepared_training_data(
        figure_of_merit="expected_fidelity"
    )
