<!-- Entries in each category are sorted by merge time, with the latest PRs appearing first. -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on a mixture of [Keep a Changelog] and [Common Changelog].
This project adheres to [Semantic Versioning], with the exception that minor releases may include breaking changes.

## [Unreleased]

### Changed

- 🔧 Replace `mypy` with `ty` ([#572]) ([**@denialhaag**])
- 🐛 Fix instruction duration unit in estimated success probability calculation ([#445]) ([**@Shaobo-Zhou**])
- ✨ Remove support for custom names of trained models ([#489]) ([**@bachase**])
- 🔥 Drop support for x86 macOS systems ([#421]) ([**@denialhaag**])

## [2.3.0] - 2025-07-29

_If you are upgrading: please see [`UPGRADING.md`](UPGRADING.md#230)._

### Added

- 📝 Add docstrings for raised errors for all methods ([#405]) ([**@nquetschlich**])
- ✨ Add Estimated Hellinger Distance as a further Figure of Merit ([#360]) ([**@flowerthrower**])

### Changed

- 🎨 Adjust the ESP reward calculation to become Qiskit v2 compatible ([#406]) ([**@nquetschlich**])
- ✨ Improve the ML part and its usability ([#403]) ([**@nquetschlich**])
- 📝 Migrate the documentation from .rst to .md files ([#403]) ([**@nquetschlich**])
- ✨ Improve RL action handling by using dataclasses ([#401]) ([**@nquetschlich**])
- ✨ Support MQT Bench v2 and use Qiskit's Target to represent quantum devices ([#393]) ([**@nquetschlich**])
- 🚚 Move to MQT organization ([#385]) ([**@flowerthrower**])

## [2.2.0] - 2025-02-02

_📚 Refer to the [GitHub Release Notes](https://github.com/munich-quantum-toolkit/predictor/releases) for previous changelogs._

<!-- Version links -->

[unreleased]: https://github.com/munich-quantum-toolkit/predictor/compare/v2.3.0...HEAD
[2.3.0]: https://github.com/munich-quantum-toolkit/predictor/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/munich-quantum-toolkit/predictor/releases/tag/v2.2.0

<!-- PR links -->

[#572]: https://github.com/munich-quantum-toolkit/predictor/pull/572
[#489]: https://github.com/munich-quantum-toolkit/predictor/pull/489
[#445]: https://github.com/munich-quantum-toolkit/predictor/pull/445
[#421]: https://github.com/munich-quantum-toolkit/predictor/pull/421
[#406]: https://github.com/munich-quantum-toolkit/predictor/pull/406
[#405]: https://github.com/munich-quantum-toolkit/predictor/pull/405
[#403]: https://github.com/munich-quantum-toolkit/predictor/pull/403
[#401]: https://github.com/munich-quantum-toolkit/predictor/pull/401
[#393]: https://github.com/munich-quantum-toolkit/predictor/pull/393
[#385]: https://github.com/munich-quantum-toolkit/predictor/pull/385
[#360]: https://github.com/munich-quantum-toolkit/predictor/pull/360

<!-- Contributor -->

[**@burgholzer**]: https://github.com/burgholzer
[**@nquetschlich**]: https://github.com/nquetschlich
[**@flowerthrower**]: https://github.com/flowerthrower
[**@denialhaag**]: https://github.com/denialhaag
[**@bachase**]: https://github.com/bachase
[**@Shaobo-Zhou**]: https://github.com/Shaobo-Zhou

<!-- General links -->

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
[Common Changelog]: https://common-changelog.org
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
[GitHub Release Notes]: https://github.com/munich-quantum-toolkit/predictor/releases
