# Welcome to MQT Predictor's documentation!

MQT Predictor is a tool for automatic device selection with device-specific circuit compilation for quantum computing.
It is part of the _{doc}`Munich Quantum Toolkit (MQT) <mqt:index>`_.

From a user's perspective, the framework works as follows:

![Illustration of the MQT Predictor framework](/_static/mqt_predictor.png)

Any uncompiled quantum circuit can be provided together with the desired figure of merit.
The framework then automatically predicts the most suitable device for the given circuit and figure of merit and compiles the circuit for the predicted device.
The compiled circuit is returned together with the compilation information and the selected device.

The MQT Predictor framework is based on two main components:

- An {doc}`Automatic Device Selection <device_selection>` component that predicts the most suitable device for a given quantum circuit and figure of merit.
- A {doc}`Device-Specific Circuit Compilation <compilation>` component that compiles a given quantum circuit for a given device.

Combining these two components, the framework can be used to automatically compile a given quantum circuit for the most suitable device optimizing a {doc}`customizable figure of merit <figure_of_merit>`.
How to install the framework is described in the {doc}`installation section <installation>`, how to set it up in the {doc}`setup section <setup>` section, and how to use it in the {doc}`quickstart section <quickstart>` section.

If you are interested in the theory behind MQT Predictor, have a look at the publications in the {doc}`references list <references>`.

---

```{toctree}
:hidden:

self
```

```{toctree}
:caption: User Guide
:glob:
:hidden:
:maxdepth: 1

installation
quickstart
setup
device_selection
compilation
figure_of_merit
references
```

```{toctree}
:caption: Developers
:glob:
:hidden:
:maxdepth: 1

contributing
development_guide
support
```

```{only} html
## Contributors and Supporters

The _[Munich Quantum Toolkit (MQT)](https://mqt.readthedocs.io)_ is developed by the [Chair for Design Automation](https://www.cda.cit.tum.de/) at the [Technical University of Munich](https://www.tum.de/) and supported by the [Munich Quantum Software Company (MQSC)](https://munichquantum.software).
Among others, it is part of the [Munich Quantum Software Stack (MQSS)](https://www.munich-quantum-valley.de/research/research-areas/mqss) ecosystem, which is being developed as part of the [Munich Quantum Valley (MQV)](https://www.munich-quantum-valley.de) initiative.

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

The MQT will remain free, open-source, and permissively licensed—now and in the future.
We are firmly committed to keeping it open and actively maintained for the quantum computing community.

To support this endeavor, please consider:

- Starring and sharing our repositories: [https://github.com/munich-quantum-toolkit](https://github.com/munich-quantum-toolkit)
- Contributing code, documentation, tests, or examples via issues and pull requests
- Citing the MQT in your publications (see {doc}`References <references>`)
- Using the MQT in research and teaching, and sharing feedback and use cases
- Sponsoring us on GitHub: [https://github.com/sponsors/munich-quantum-toolkit](https://github.com/sponsors/munich-quantum-toolkit)

<p align="center">
<iframe src="https://github.com/sponsors/munich-quantum-toolkit/button" title="Sponsor munich-quantum-toolkit" height="32" width="114" style="border: 0; border-radius: 6px;"></iframe>
</p>
```
