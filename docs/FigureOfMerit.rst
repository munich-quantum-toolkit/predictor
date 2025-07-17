Figure of Merit
================

The figure of merit determines the optimization criteria for both the ML and RL model and, thus, is a crucial part of the MQT Predictor framework.
In general, it can be completely customizable and consider anything from (compiled)
circuit characteristics such as gate counts to actual execution costs by a specific vendor. Nevertheless,
a comprehensive figure of merit should consider at least the characteristics of the compiled quantum
circuit and the characteristics of the selected quantum device. Therefore, the respective hardware
information needs to be collected, such as, e.g., gate/readout fidelities, gate execution times, or
decoherence times. While for some devices, this information may be publicly available, for other
devices it may be estimated from comparable devices, previous records, or insider knowledge.

So far, two figures of merit are implemented for all devices: ``expected_fidelity`` and ``critical_depth``.
On top of that, the ``estimated_success_probability`` is available for devices where the necessary calibration data is available, as well as the ``estimated_hellinger_distance`` when a suitable (trained) model is provided.

Expected Fidelity
-----------------

Consider the figure of merit that takes into account the following two aspects:

- If the selected device is not large enough to execute a given quantum circuit with respective to its number of qubits, the worst-possible score is assigned.
- If the device is large enough, the evaluation score is calculated using the formula:

.. math::
    \mathit{\mathcal{F}}=\prod_{i=1}^{|G|} \mathit{\mathcal{F}}(g_i) \prod_{j=1}^{m} \mathit{\mathcal{F}_{RO}}(q_j)

with :math:`\mathit{\mathcal{F}}(g_i)` being the expected execution fidelity of gate :math:`g_i` on its corresponding qubit(s),
:math:`\mathit{\mathcal{F}_{RO}}(q_j)` being the expected execution fidelity of a measurement operation :math:`q_j` on its corresponding qubit and :math:`|G|` respectively :math:`m` being the number of gates and measurements in the compiled circuit.


This figure of merit determines an estimate of the probability that a quantum circuit will return the expected result, the so-called ``expected fidelity``, which ranges between :math:`0.0` and :math:`1.0` with higher values being better.


Critical Depth
--------------
A potential alternative could be the ``critical depth`` (taken from :cite:labelpar:`tomesh2022supermarq`)---a measure to describe the percentage of multi-qubit gates on the longest path through a compiled quantum circuit (determining the depth).
A respective value close to 1 would indicate a very sequential circuit while a value of 0 would indicate a highly parallel one.


Estimated Success Probability
-----------------------------
The ``estimated_success_probability`` (based on :cite:labelpar:`esp-lifetime-min`, :cite:labelpar:`esp-lifetime`, and :cite:labelpar:`esp-idle`) is a figure of merit that is based on the ``expected_fidelity`` but also multiplies it with a factor that considers the decoherence times :math:`T_1, T_2` of a device:

.. math::
   \prod_{q} \exp{-(t_{q}^{\mathrm{idle}}/\min{(T_1, T_2)})}

with :math:`t_{q}^{\mathrm{idle}}` being the sum of each qubit's idle times.
Therefore, exactly the execution times of all gates and the decoherence times must be available.
Note that some variants of this figure of merit do not take the minimum of both decoherence times but create one exponential factor for each decoherence time, others consider the entire qubit lifetime instead of idle times only.


Estimated Hellinger Distance
----------------------------
The ``estimated_hellinger_distance`` is a figure of merit based on a machine learning (ML) model that has been trained on a dataset of quantum circuits labeled with their respective Hellinger distance.

To use this figure of merit, three steps are required:

1. **Feature Extraction:** Prepare a set of compiled quantum circuits (e.g., from ``MQT Bench``) for execution on a target device and extract corresponding feature vectors.

   .. code-block:: python

      from mqt.predictor.hellinger import calc_device_specific_features

      feature_vector_list = []
      for qc in quantum_circuits:
          feature_vector = calc_device_specific_features(qc, device)
          feature_vector_list.append(feature_vector)

2. **Label Generation:** Compute the Hellinger distance between the noisy probability distribution (obtained from executing on a quantum device) and the noiseless distribution (from simulation, e.g., using ``MQT DDSIM``).

   .. code-block:: python

      from mqt.predictor.hellinger import hellinger_distance
      from mqt.predictor.ml import TrainingData

      labels_list = []
      for noisy, noiseless in zip(noisy_distributions, noiseless_distributions):
          distance_label = hellinger_distance(noisy, noiseless)
          labels_list.append(distance_label)

      training_data = ml.helper.TrainingData(X_train=feature_vector_list, y_train=labels_list)

3. **Model Training:** Train an ML model using the compiled quantum circuit features and the Hellinger distance labels.

   .. code-block:: python

      from mqt.predictor.ml import train_random_forest_model, Predictor

      pred = Predictor(
          figure_of_merit="hellinger_distance", devices=[get_device("ibm_falcon_27")]
      )
      pred.train_random_forest_model(training_data)

Once the model has been successfully trained, the ``estimated_hellinger_distance`` figure of merit can serve as a device-specific figure of merit to assess the quality of a compiled quantum circuit (i.e. calculate a Hellinger distance value :math:`\in [0, 1])`).

   .. code-block:: python

      from mqt.predictor.reward import estimated_hellinger_distance

      print(estimated_hellinger_distance(quantum_circuits[0], device, trained_model))

In the context of the MQT Predictor, it can be used as a reward function in the RL module and subsequently utilized in the ML module to score and compare quantum devices, just like any other figure of merit mentioned above.
