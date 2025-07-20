# Welcome to MQT Predictor's documentation!

MQT Predictor is a tool for Automatic Device Selection with Device-Specific Circuit Compilation for Quantum Computing developed as part of the [Munich Quantum Toolkit](https://mqt.readthedocs.io) (_MQT_).

From a user's perspective, the framework works as follows:

![Illustration of the MQT Predictor framework](/_static/mqt_predictor.png){ width=100% }

Any uncompiled quantum circuit can be provided together with the desired figure of merit.  
The framework then automatically predicts the most suitable device for the given circuit and figure of merit and compiles the circuit for the predicted device.  
The compiled circuit is returned together with the compilation information and the selected device.

The MQT Predictor framework is based on two main components:

- An [Automatic Device Selection](device_selection.md) component that predicts the most suitable device for a given quantum circuit and figure of merit.
- A [Device-Specific Circuit Compilation](compilation.md) component that compiles a given quantum circuit for a given device.

Combining these two components, the framework can be used to automatically compile a given quantum circuit for the most suitable device optimizing a [customizable figure of merit](figure_of_merit.md).
How to use the framework is described in the [usage](usage.md) section.

If you are interested in the theory behind MQT Predictor, have a look at the publications in the [references list](references.md).

---

```{toctree}
:hidden: true

self
```

```{toctree}
:caption: User Guide
:glob: true
:maxdepth: 1

quickstart
usage
device_selection
compilation
figure_of_Merit
references
```

```{toctree}
:caption: Developers
:glob: true
:maxdepth: 1

contributing
development_guide
support
```

````{only} html
## Contributors and Supporters
The _[Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io)_ is developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/)
and supported by the [Munich Quantum Software Company (MQSC)](https://munichquantum.software).
Among others, it is part of the [Munich Quantum Software Stack (MQSS)](https://www.munich-quantum-valley.de/research/research-areas/mqss) ecosystem,
which is being developed as part of the [Munich Quantum Valley (MQV)](https://www.munich-quantum-valley.de) initiative.
<div style="margin-top: 0.5em">
<div class="only-light" align="center">
  <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-light.svg" width="90%" alt="MQT Banner">
</div>
<div class="only-dark" align="center">
  <img src="https://raw.githubusercontent.com/munich-quantum-toolkit/.github/refs/heads/main/docs/_static/mqt-logo-banner-dark.svg" width="90%" alt="MQT Banner">
</div>
</div>
Thank you to all the contributors who have helped make MQT Predictor a reality!
<p align="center">
<a href="https://github.com/munich-quantum-toolkit/predictor/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=munich-quantum-toolkit/predictor" />
</a>
</p>

## Cite This

If you want to cite MQT Predictor, please use the following BibTeX entry:

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
````