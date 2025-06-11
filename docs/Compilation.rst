Device-specific Compilation
===========================
Once a target device has been selected, the quantum circuit, which is typically designed in
a device-agnostic fashion that does not account for any hardware limitations (such as a limited
gate-set or limited connectivity), must be compiled accordingly so that it actually becomes executable
on that device.

Compilation itself is a sequential process consisting of a sequence of compilation passes that, step-by-step, transform the original quantum circuit so that it
eventually conforms to the limitations imposed by the target device. Since many of the underlying
problems in compilation are computationally hard, there is an ever-growing
variety of compilation passes available across several quantum SDKs and software tools—again, each
with their own advantages and disadvantages.

As a result of the sheer number of options, choosing the best sequence of compilation passes for a given
application is nearly impossible. Consequently, most quantum SDKs (such as Qiskit and TKET) provide
easy-to-use high-level function calls that
encapsulate "their" sequence of compilation passes into a single compilation flow. While this allows
to conveniently compile circuits, it has several drawbacks:

- It creates a kind of vendor lock that limits the available compilation passes to those available in the SDK offering the compilation flow.
- The respective compilation flows are designed to be broadly applicable and, hence, are neither device-specific nor circuit-specific.
- No means are provided to optimize for a customizable figure of merit.


To overcome these limitations, the MQT Predictor framework provides device-specific
quantum circuit compilers by combining compilation passes from various compiler tools
and learning optimized sequences of those passes with respect to a customizable figure of
merit). This mix-and-match of compiler passes from various tools allows one to eliminate
vendor locks and to create optimized compilers that transcend the individual tools.


The compilation process is modelled as a Markov Decision Process and takes three inputs to train a respective reinforcement learning (RL) model that acts as a compiler:

1. Training circuits
2. The targeted quantum device
3. The figure of merit to optimize for


.. image:: /_static/rl.png
   :width: 100%
   :alt: Illustration of the RL model
   :align: center

The trained model can be used to compile any quantum circuit for the targeted device.

In this implementation, compilation passes from both IBM's Qiskit and Quantinuum's TKET are utilized for the RL training
of the optimized compiler.
We trained one RL model for each currently :ref:`supported quantum device <supported-quantum-devices>`.



Training Data
-------------
To train the model, sufficient training data must be provided as qasm files in the `respective directory <https://github.com/munich-quantum-toolkit/predictor/tree/main/src/mqt/predictor/rl/training_data/training_circuits>`_.
We provide the training data used for the initial performance evaluation of this framework which are stored `here <https://github.com/munich-quantum-toolkit/predictor/tree/main/src/mqt/predictor/rl/training_data/trained_model>`_.
