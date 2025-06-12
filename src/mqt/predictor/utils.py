# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utility functions for the mqt.predictor module."""

from __future__ import annotations

import logging
import signal
import sys
from typing import TYPE_CHECKING, Any
from warnings import warn

if TYPE_CHECKING:
    from collections.abc import Callable

    from qiskit import QuantumCircuit

    from mqt.predictor.reward import figure_of_merit
    from mqt.predictor.rl.predictor import Predictor as RL_Predictor

logger = logging.getLogger("mqt-predictor")


def timeout_watcher(
    func: Callable[..., bool | QuantumCircuit],
    args: list[QuantumCircuit | figure_of_merit | str | RL_Predictor],
    timeout: int,
) -> tuple[QuantumCircuit, list[str]] | bool:
    """Method that stops a function call after a given timeout limit."""
    if sys.platform == "win32":
        warn("Timeout is not supported on Windows.", category=RuntimeWarning, stacklevel=2)
        return func(*args) if isinstance(args, tuple | list) else func(args)

    class TimeoutExceptionError(Exception):  # Custom exception class
        pass

    def timeout_handler(_signum: int, _frame: Any) -> None:  # noqa: ANN401
        raise TimeoutExceptionError

    # Change the behavior of SIGALRM
    signal.signal(signal.SIGALRM, timeout_handler)

    signal.alarm(timeout)
    try:
        res = func(*args)
    except TimeoutExceptionError:
        logger.debug("Calculation/Generation exceeded timeout limit for " + func.__module__ + ", " + str(args[1:]))
        return False
    except Exception:
        logger.exception("Something else went wrong")
        return False
    else:
        # Reset the alarm
        signal.alarm(0)

    return res
