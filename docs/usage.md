# Usage

There are two ways how to use MQT Predictor:

1. Via the pip package `mqt.predictor`
2. Directly via this repository

## Usage via pip package

MQT Predictor is available via [PyPI](https://pypi.org/project/mqt.predictor/)

    > ```console
    > (venv) $ pip install mqt.predictor
    > ```

To compile a quantum circuit, use the `qcompile` method:

.. automodule:: mqt.predictor
:members: qcompile

Currently available figures of merit are:

```{code-cell} ipython3
:tags: [hide-input]
from mqt.predictor.reward import figures_of_merit
print(figures_of_merit)
```

An example of how `qcompile` is used can be found in the [Quickstart](Quickstart) Jupyter notebook.

## Usage directly via this repository

For that, the repository must be cloned and installed:

    > ```console
    > git clone https://github.com/munich-quantum-toolkit/predictor.git
    > cd mqt-predictor
    > pip install .
    > ```

Afterwards, the package can be used as described above.
