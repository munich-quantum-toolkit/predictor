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

# Automatic Device Selection

To realize any quantum application, a suitable quantum device must be selected for the execution of the developed quantum algorithm.
This alone is non-trivial since new quantum devices based on various underlying technologies emerge on an almost daily basis—each with their own advantages and disadvantages.
There are hardly any practical guidelines on which device to choose based on the targeted application.
As such, the best guess in many cases today is to simply try out many (if not all) possible devices and, afterwards, choose the best results—certainly a time- and resource-consuming endeavor that is not sustainable for the future.

A naive approach to select the best quantum device for a given quantum circuit would be to compile it for all devices, e.g., using the trained RL models which act as specialized compilers for supported quantum devices.
Afterwards, the resulting compiled circuits must be evaluated according to some figure of merit to identify the most promising device.
However, doing this for each and every to-be-compiled quantum circuit is practically infeasible since compilation is a time-consuming task.

The MQT Predictor framework provides an easy-to-use solution to this problem by using supervised machine learning.
It learns from previous compilations of other quantum circuits and models the problem of determining the most promising device for a circuit and figure of merit as a statistical classification task — a task well suited for supervised machine learning.
For that, the framework is trained with based on three inputs:

1. Training circuits
2. The compilation options for all supported devices
3. The figure of merit to optimize for

![Illustration of the ML model](/_static/ml.png)

The trained model then acts as a predictor and can be used to predict the most suitable device for a given quantum circuit and figure of merit.

## Supported Quantum Devices

Any device provided as a Qiskit `Target` object can be used with the MQT Predictor framework.
MQT Bench provides a set of devices that can be used out-of-the-box which are available under
[MQT Bench Devices and Parameters](https://mqt.readthedocs.io/projects/bench/en/latest/parameter.html).
Currently, the following devices are supported:
So far, MQT Bench supports the following devices:

```{code-cell} ipython3
:tags: [hide-input]
from mqt.bench.targets import get_device, get_available_device_names

for num, device_name in enumerate(get_available_device_names()):
    print(f"{num+1}: {device_name} with {get_device(device_name).num_qubits} qubits")
```

Adding further devices is straightforward and requires only to provide its native gate-set, connectivity, and calibration data.
