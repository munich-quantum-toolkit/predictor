[![PyPI](https://img.shields.io/pypi/v/mqt.predictor?logo=pypi&style=flat-square)](https://pypi.org/project/mqt.predictor/)
![OS](https://img.shields.io/badge/os-linux%20%7C%20macos%20%7C%20windows-blue?style=flat-square)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/JOSS-10.21105/joss.07478-blue.svg?style=flat-square)](https://doi.org/10.21105/joss.07478)
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

# MQT Predictor - Automatic Device Selection with Device-Specific Circuit Compilation for Quantum Computing

MQT Predictor is a framework that allows one to automatically select a suitable quantum device for a particular application and provides an optimized compiler for the selected device.
It is part of the [_Munich Quantum Toolkit (MQT)_](https://mqt.readthedocs.io).

<p align="center">
  <a href="https://mqt.readthedocs.io/projects/predictor">
  <img width=30% src="https://img.shields.io/badge/documentation-blue?style=for-the-badge&logo=read%20the%20docs" alt="Documentation" />
  </a>
</p>

## Key Features

MQT Predictor supports end-users in navigating the vast landscape of choices by allowing them to mix-and-match compiler passes from various tools to create optimized compilers that transcend the individual tools.
Evaluations on more than 500 quantum circuits and seven devices have shown that—compared to Qiskit's and TKET's most optimized compilation flows—the MQT Predictor yields circuits with an expected fidelity that is on par with the best possible result that could be achieved by trying out all combinations of devices and compilers and even achieves a similar performance when considering the critical depth as an alternative figure of merit.

Therefore, MQT Predictor tackles this problem from two angles:

1. It provides a method (based on Reinforcement Learning) that produces device-specific quantum circuit compilers by combining compilation passes from various compiler tools and learning optimized sequences of those passes with respect to a customizable figure of merit.
   This mix-and-match of compiler passes from various tools allows one to eliminate vendor locks and to create optimized compilers that transcend the individual tools.

2. It provides a prediction method (based on Supervised Machine Learning) that, without performing any compilation, automatically predicts the most suitable device for a given application.
   This completely eliminates the manual and laborious task of determining a suitable target device and guides end-users through the vast landscape of choices without the need for quantum computing expertise.

<p align="center">
<picture>
  <img src="docs/_static/problem.png" width="100%">
</picture>
</p>

If you have any questions, feel free to create a [discussion](https://github.com/munich-quantum-toolkit/predictor/discussions) or an [issue](https://github.com/munich-quantum-toolkit/predictor/issues) on [GitHub](https://github.com/munich-quantum-toolkit/predictor).

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
  <img src="https://contrib.rocks/image?repo=munich-quantum-toolkit/predictor" alt="Contributors to munich-quantum-toolkit/predictor" />
  </a>
</p>

The MQT will remain free, open-source, and permissively licensed—now and in the future.
We are firmly committed to keeping it open and actively maintained for the quantum computing community.

To support this endeavor, please consider:

- Starring and sharing our repositories: https://github.com/munich-quantum-toolkit
- Contributing code, documentation, tests, or examples via issues and pull requests
- Citing the MQT in your publications (see [Cite This](#cite-this))
- Citing our research in your publications (see [References](https://mqt.readthedocs.io/projects/predictor/en/latest/references.html))
- Using the MQT in research and teaching, and sharing feedback and use cases
- Sponsoring us on GitHub: https://github.com/sponsors/munich-quantum-toolkit

<p align="center">
  <a href="https://github.com/sponsors/munich-quantum-toolkit">
  <img width=20% src="https://img.shields.io/badge/Sponsor-white?style=for-the-badge&logo=githubsponsors&labelColor=black&color=blue" alt="Sponsor the MQT" />
  </a>
</p>

## Getting Started

`mqt.predictor` is available via [PyPI](https://pypi.org/project/mqt.predictor/).

```console
(.venv) $ pip install mqt.predictor
```

The following code gives an example on the usage:

```python3
from mqt.predictor import qcompile
from mqt.bench import get_benchmark, BenchmarkLevel

# Get a benchmark circuit from MQT Bench
qc_uncompiled = get_benchmark(benchmark="ghz", level=BenchmarkLevel.ALG, circuit_size=5)

# Compile it using the MQT Predictor
qc_compiled, compilation_information, quantum_device = qcompile(
    qc=qc_uncompiled,
    figure_of_merit="expected_fidelity",
)

# Print the selected device and the compilation information
print(quantum_device, compilation_information)

# Draw the compiled circuit
print(qc_compiled.draw())
```

> [!NOTE]
> To execute the code, respective machine learning models must be trained before.
> Up until mqt.predictor v2.0.0, pre-trained models were provided.
> However, this is not feasible anymore due to the increasing number of devices and figures of merits.
> Instead, we now provide a detailed documentation on how to train and setup the MQT Predictor framework.

**Detailed documentation and examples are available at [ReadTheDocs](https://mqt.readthedocs.io/projects/predictor).**

## System Requirements

MQT Predictor can be installed on all major operating systems with all supported Python versions.
Building (and running) is continuously tested under Linux, macOS, and Windows using the [latest available system versions for GitHub Actions](https://github.com/actions/runner-images).

## Cite This

Please cite the work that best fits your use case.

### MQT Predictor (the tool)

When citing the software itself or results produced with it, cite the MQT Predictor paper:

```bibtex
@article{quetschlich2025mqtpredictor,
  title        = {{MQT Predictor: Automatic Device Selection with Device-Specific Circuit Compilation for Quantum Computing}},
  author       = {Quetschlich, Nils and Burgholzer, Lukas and Wille, Robert},
  year         = {2025},
  journal      = {ACM Transactions on Quantum Computing (TQC)},
  doi          = {10.1145/3673241},
  eprint       = {2310.06889},
  eprinttype   = {arxiv}
}
```

### The Munich Quantum Toolkit (the project)

When discussing the overall MQT project or its ecosystem, cite the MQT Handbook:

```bibtex
@inproceedings{mqt,
  title        = {The {{MQT}} Handbook: {{A}} Summary of Design Automation Tools and Software for Quantum Computing},
  shorttitle   = {{The MQT Handbook}},
  author       = {Wille, Robert and Berent, Lucas and Forster, Tobias and Kunasaikaran, Jagatheesan and Mato, Kevin and Peham, Tom and Quetschlich, Nils and Rovara, Damian and Sander, Aaron and Schmid, Ludwig and Schoenberger, Daniel and Stade, Yannick and Burgholzer, Lukas},
  year         = 2024,
  booktitle    = {IEEE International Conference on Quantum Software (QSW)},
  doi          = {10.1109/QSW62656.2024.00013},
  eprint       = {2405.17543},
  eprinttype   = {arxiv},
  addendum     = {A live version of this document is available at \url{https://mqt.readthedocs.io}}
}
```

---

## Acknowledgements

The Munich Quantum Toolkit has been supported by the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement No. 101001318), the Bavarian State Ministry for Science and Arts through the Distinguished Professorship Program, as well as the Munich Quantum Valley, which is supported by the Bavarian state government with funds from the Hightech Agenda Bayern Plus.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-dark.svg" width="90%">
    <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-funding-footer-light.svg" width="90%" alt="MQT Funding Footer">
  </picture>
</p>
