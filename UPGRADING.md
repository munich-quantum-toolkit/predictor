# Upgrade Guide

This document describes breaking changes and how to upgrade. For a complete list of changes including minor and patch releases, please refer to the [changelog](CHANGELOG.md).

## [Unreleased]

With this release, `mqt-predictor` no longer supports the x86 architecture on macOS.
This step was necessary to ensure compatibility with PyTorch.

## [2.3.0] - 2025-07-29

In this release, we have migrated to using Qiskit's `Target` class to represent quantum devices.
This change allows for better compatibility with the latest MQT Bench version and improves the overall usability of the library.
Beyond that, we also support Qiskit v2 now.

Furthermore, both the ML and RL parts of MQT Predictor have been refactored to enhance their functionality and usability:
The ML setup has been simplified and streamlined, making it easier to use and integrate into your workflows.
The RL action handling has been updated to utilize dataclasses, which improves the structure and clarity of the code, making it easier to understand and maintain.

### General

MQT Bench has moved to the [munich-quantum-toolkit](https://github.com/munich-quantum-toolkit) GitHub organization under https://github.com/munich-quantum-toolkit/predictor.
While most links should be automatically redirected, please update any links in your code to point to the new location.
All links in the documentation have been updated accordingly.

<!-- Version links -->

[unreleased]: https://github.com/munich-quantum-toolkit/predictor/compare/v2.3.0...HEAD
[2.3.0]: https://github.com/munich-quantum-toolkit/predictor/compare/v2.2.0...v2.3.0
