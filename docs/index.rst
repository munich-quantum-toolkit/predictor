Welcome to MQT Predictor's documentation!
=========================================

MQT Predictor is a tool for Automatic Device Selection with Device-Specific Circuit Compilation for Quantum Computing developed as part of the `Munich Quantum Toolkit <https://mqt.readthedocs.io>`_ (*MQT*) by the `Chair for Design Automation <https://www.cda.cit.tum.de/>`_ at the `Technical University of Munich <https://www.tum.de>`_.

From a user's perspective, the framework is used as follows:

.. image:: /_static/mqt_predictor.png
   :width: 100%
   :alt: Illustration of the MQT Predictor framework
   :align: center

Any uncompiled quantum circuit can be provided together with the desired figure of merit.
The framework then automatically predicts the most suitable device for the given circuit and figure of merit and compiles the circuit for the predicted device.
The compiled circuit is returned together with the compilation information and the selected device.

The MQT Predictor framework is based on two main components:

- An :doc:`Automatic Device Selection <DeviceSelection>` component that predicts the most suitable device for a given quantum circuit and figure of merit.
- A :doc:`Device-Specific Circuit Compilation <Compilation>` component that compiles a given quantum circuit for a given device.

Combining these two components, the framework can be used to automatically compile a given quantum circuit for the most suitable device optimizing a :doc:`customizable figure of merit<FigureOfMerit>`.
How to use the framework is described in the :doc:`Usage <Usage>` section.

If you are interested in the theory behind MQT Predictor, have a look at the publications in the :doc:`references list <References>`.


----

 .. toctree::
    :hidden:

    self

 .. toctree::
    :maxdepth: 1
    :caption: User Guide
    :glob:

    Quickstart
    Usage
    Setup
    DeviceSelection
    Compilation
    FigureOfMerit
    References

 .. toctree::
    :maxdepth: 1
    :caption: Developers
    :glob:

    Contributing
    DevelopmentGuide
    Support
