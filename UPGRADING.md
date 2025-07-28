# Upgrade Guide

This document describes breaking changes and how to upgrade. For a complete list of changes including minor and patch releases, please refer to the [changelog](CHANGELOG.md).

## [Unreleased]

## [2.3.0] - 2025-07-29

In this release, we have migrated to using Qiskit's `Target` class to represent quantum devices. This change allows for better compatibility with the latest MQT Bench version and improves the overall usability of the library.

Furthermore, both the ML and RL parts of MQT Predictor have been refactored to enhance their functionality and usability:
The ML setup has been simplified and streamlined, making it easier to use and integrate into your workflows.
The RL action handling has been updated to utilize dataclasses, which improves the structure and clarity of the code, making it easier to understand and maintain.

<!-- Version links -->

[unreleased]: https://github.com/munich-quantum-toolkit/predictor/compare/v2.3.0...HEAD
[2.3.0]: https://github.com/munich-quantum-toolkit/predictor/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/munich-quantum-toolkit/predictor/releases/tag/v2.2.0
