[![PyPI](https://img.shields.io/pypi/v/mqt.predictor?logo=pypi&style=flat-square)](https://pypi.org/project/mqt.predictor/)
![OS](https://img.shields.io/badge/os-linux%20%7C%20macos%20%7C%20windows-blue?style=flat-square)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![CI](https://img.shields.io/github/actions/workflow/status/munich-quantum-toolkit/predictor/ci.yml?branch=main&style=flat-square&logo=github&label=ci)](https://github.com/munich-quantum-toolkit/predictor/actions/workflows/ci.yml)
[![CD](https://img.shields.io/github/actions/workflow/status/munich-quantum-toolkit/predictor/cd.yml?style=flat-square&logo=github&label=cd)](https://github.com/munich-quantum-toolkit/predictor/actions/workflows/cd.yml)
[![Documentation](https://img.shields.io/readthedocs/mqt-predictor?logo=readthedocs&style=flat-square)](https://mqt.readthedocs.io/projects/predictor)
[![codecov](https://img.shields.io/codecov/c/github/munich-quantum-toolkit/predictor?style=flat-square&logo=codecov)](https://codecov.io/gh/munich-quantum-toolkit/predictor)

<p align="center">
  <a href="https://mqt.readthedocs.io">
   <picture>
     <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/logo-mqt-dark.svg" width="60%">
     <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/logo-mqt-light.svg" width="60%" alt="MQT Logo">
   </picture>
  </a>
</p>

# MQT Predictor: Automatic Device Selection with Device-Specific Circuit Compilation for Quantum Computing

MQT Predictor is a framework that allows one to automatically select a suitable quantum device for a particular application and provides an optimized compiler for the selected device.
It not only supports end-users in navigating the vast landscape of choices, it also allows to mix-and-match compiler passes from various tools to create optimized compilers that transcend the individual tools.
Evaluations on more than 500 quantum circuits and seven devices have shown that—compared to Qiskit's and TKET's most optimized compilation flows—the MQT Predictor yields circuits with an expected fidelity that is on par with the best possible result that could be achieved by trying out all combinations of devices and compilers and even achieves a similar performance when considering the critical depth as an alternative figure of merit.

Therefore, MQT Predictor tackles this problem from two angles:

1. It provides a method (based on Reinforcement Learning) that produces device-specific quantum circuit compilers by combining compilation passes from various compiler tools and learning optimized sequences of those passes with respect to a customizable figure of merit). This mix-and-match of compiler passes from various tools allows one to eliminate vendor locks and to create optimized compilers that transcend the individual tools.

2. It provides a prediction method (based on Supervised Machine Learning) that, without performing any compilation, automatically predicts the most suitable device for a given application. This completely eliminates the manual and laborious task of determining a suitable target device and guides end-users through the vast landscape of choices without the need for quantum computing expertise.

<p align="center">
<picture>
  <img src="docs/_static/problem.png" width="100%">
</picture>
</p>

For more details, please refer to:

<p align="center">
  <a href="https://mqt.readthedocs.io/projects/predictor">
  <img width=30% src="https://img.shields.io/badge/documentation-blue?style=for-the-badge&logo=read%20the%20docs" alt="Documentation" />
  </a>
</p>

## Contributors and Supporters

The _[Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io)_ is developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/) and supported by the [Munich Quantum Software Company (MQSC)](https://munichquantum.software).
Among others, it is part of the [Munich Quantum Software Stack (MQSS)](https://www.munich-quantum-valley.de/research/research-areas/mqss) ecosystem, which is being developed as part of the [Munich Quantum Valley (MQV)](https://www.munich-quantum-valley.de) initiative.

<p align="center">
  <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-dark.svg" width="90%">
   <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-light.svg" width="90%" alt="MQT Partner Logos">
  </picture>
</p>

Thank you to all the contributors who have helped make MQT Predictor a reality!

<p align="center">
<a href="https://github.com/munich-quantum-toolkit/predictor/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=munich-quantum-toolkit/predictor" />
</a>
</p>

## Getting Started

`mqt.predictor` is available via [PyPI](https://pypi.org/project/mqt.predictor/).

```console
(venv) $ pip install mqt.predictor
```

The following code gives an example on the usage:

```python3
from mqt.predictor import qcompile
from mqt.bench import get_benchmark

# get a benchmark circuit on algorithmic level representing the GHZ state with 5 qubits from [MQT Bench](https://github.com/munich-quantum-toolkit/bench)
qc_uncompiled = get_benchmark(benchmark_name="ghz", level="alg", circuit_size=5)

# compile it using the MQT Predictor
qc_compiled, compilation_information, quantum_device = qcompile(
    qc_uncompiled, figure_of_merit="expected_fidelity"
)

# print the selected device and the compilation information
print(quantum_device, compilation_information)

# draw the compiled circuit
print(qc_compiled.draw())
```

> [!NOTE]
> To execute the code, respective machine learning models must be trained before.
> Up until mqt.predictor v2.0.0, pre-trained models were provided. However, this is not feasible anymore due to the
> increasing number of devices and figures of merits. Instead, we now provide a detailed documentation on how to train
> and setup the MQT Predictor framework.\*\*

**Further documentation and examples are available at [ReadTheDocs](https://mqt.readthedocs.io/projects/predictor).**

## References

In case you are using MQT Predictor in your work, we would be thankful if you referred to it by citing the following publication:

```bibtex
@ARTICLE{quetschlich2025mqtpredictor,
    AUTHOR      = {N. Quetschlich and L. Burgholzer and R. Wille},
    TITLE       = {{MQT Predictor: Automatic Device Selection with Device-Specific Circuit Compilation for Quantum Computing}},
    YEAR        = {2025},
    JOURNAL     = {ACM Transactions on Quantum Computing (TQC)},
    DOI         = {10.1145/3673241},
    EPRINT      = {2310.06889},
    EPRINTTYPE  = {arxiv},
}
```

---

## Acknowledgements

This project received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research
and innovation program (grant agreement No. 101001318), was part of the Munich Quantum Valley, which is supported by the
Bavarian state government with funds from the Hightech Agenda Bayern Plus, and has been supported by the BMWK on the
basis of a decision by the German Bundestag through project QuaST, as well as by the BMK, BMDW, the State of Upper
Austria in the frame of the COMET program, and the QuantumReady project within Quantum Austria (managed by the FFG).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-dark.svg" width="90%">
    <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-light.svg" width="90%" alt="MQT Funding Footer">
  </picture>
</p>
