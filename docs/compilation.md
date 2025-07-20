# Device-specific Compilation

Once a target device has been selected, the quantum circuit — typically designed in a device-agnostic fashion that does not account for hardware limitations (such as a limited gate set or limited connectivity) — must be compiled so that it actually becomes executable on that device.

Compilation itself is a sequential process consisting of a series of compilation passes that step-by-step transform the original quantum circuit until it conforms to the limitations imposed by the target device. Since many of the underlying problems in compilation are computationally hard, an ever-growing variety of compilation passes is available across several quantum SDKs and software tools—each with their own advantages and disadvantages.

As a result of the sheer number of options, choosing the best sequence of compilation passes for a given application is nearly impossible. Consequently, most quantum SDKs (such as Qiskit and TKET) provide easy-to-use high-level function calls that encapsulate *their* sequence of compilation passes into a single compilation flow. While this allows convenient circuit compilation, it has several drawbacks:

- It creates a kind of vendor lock that limits the available compilation passes to those provided by the SDK offering the compilation flow.
- The respective compilation flows are designed to be broadly applicable and are neither device-specific nor circuit-specific.
- No means are provided to optimize for a customizable figure of merit.

To overcome these limitations, the MQT Predictor framework provides **device-specific quantum circuit compilers** by combining compilation passes from various compiler tools and learning optimized sequences of those passes with respect to a customizable figure of merit. This mix-and-match approach eliminates vendor lock-in and enables optimized compilers that transcend the capabilities of individual tools.

The compilation process is modeled as a **Markov Decision Process** and takes three inputs to train a reinforcement learning (RL) model that acts as a compiler:

1. Training circuits  
2. The targeted quantum device  
3. The figure of merit to optimize for  

![Illustration of the RL model](/_static/rl.png)

The trained model can then be used to compile any quantum circuit for the targeted device.

---

## Training Data

To train the model, sufficient training data must be provided as QASM files in the [respective directory](https://github.com/munich-quantum-toolkit/predictor/tree/main/src/mqt/predictor/rl/training_data/training_circuits).