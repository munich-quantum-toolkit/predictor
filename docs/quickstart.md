---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

```{code-cell} ipython3
:tags: [remove-cell]
%config InlineBackend.figure_formats = ['svg']
```

# Quickstart

```{code-cell} ipython3
from mqt.bench import BenchmarkLevel, get_benchmark

from mqt.predictor import qcompile
```

## Get Uncompiled Quantum Circuit
```{code-cell} ipython3
qc = get_benchmark("ghz", level=BenchmarkLevel.INDEP, circuit_size=5)
qc.draw()
```

## Compile using MQT Predictor
```{code-cell} ipython3
qc_compiled, compilation_information, quantum_device = qcompile(qc, figure_of_merit="expected_fidelity")
```

## Predicted Device
```{code-cell} ipython3
print(quantum_device)
```

## Executed Compilation Passes
```{code-cell} ipython3
print(compilation_information)
```

## Compiled Circuit
```{code-cell} ipython3
qc_compiled.draw()
```