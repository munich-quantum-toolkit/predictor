#!/usr/bin/env -S uv run --script --quiet
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# /// script
# dependencies = ["nox"]
# ///

"""Nox sessions."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import nox

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


nox.needs_version = ">=2025.10.16"
nox.options.default_venv_backend = "uv"

PYTHON_ALL_VERSIONS = ["3.11", "3.12", "3.13", "3.14"]
PYTHON_LATEST_VERSION = PYTHON_ALL_VERSIONS[-1]


def _is_draft_pull_request() -> bool:
    """Determine whether GitHub Actions is running for a draft pull request."""
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if os.environ.get("GITHUB_EVENT_NAME") != "pull_request" or event_path is None:
        return False

    with Path(event_path).open(encoding="utf-8") as event_file:
        event = json.load(event_file)
    return bool(event.get("pull_request", {}).get("draft", False))


IS_DRAFT_PULL_REQUEST = _is_draft_pull_request()
PYTHON_TEST_VERSIONS = PYTHON_ALL_VERSIONS[-1:] if IS_DRAFT_PULL_REQUEST else PYTHON_ALL_VERSIONS

RL_MODEL_TESTS = (
    "tests/compilation/test_predictor_rl.py::test_qcompile_with_newly_trained_models",
    "tests/compilation/test_predictor_rl.py::test_qcompile_generates_trace_file",
    "tests/hellinger_distance/test_estimated_hellinger_distance.py::test_train_and_qcompile_with_hellinger_model",
)

if os.environ.get("CI", None):
    nox.options.error_on_missing_interpreters = True


@contextlib.contextmanager
def preserve_lockfile() -> Generator[None]:
    """Preserve the lockfile by moving it to a temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir_name:
        shutil.move("uv.lock", f"{temp_dir_name}/uv.lock")
        try:
            yield
        finally:
            shutil.move(f"{temp_dir_name}/uv.lock", "uv.lock")


@nox.session(reuse_venv=True, default=True)
def lint(session: nox.Session) -> None:
    """Run the linter."""
    if shutil.which("prek") is None:
        session.install("prek")

    session.run("prek", "run", "--all-files", *session.posargs, external=True)


def _run_tests(
    session: nox.Session,
    *,
    install_args: Sequence[str] = (),
    extra_command: Sequence[str] = (),
    pytest_run_args: Sequence[str] = (),
    run_rl_training: bool = False,
) -> None:
    env = {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}

    if os.environ.get("GITHUB_ACTIONS") == "true" and not (
        run_rl_training
        and not IS_DRAFT_PULL_REQUEST
        and os.environ.get("RUNNER_OS") == "Linux"
        and session.python == PYTHON_LATEST_VERSION
    ):
        pytest_run_args = (*pytest_run_args, *(f"--deselect={test}" for test in RL_MODEL_TESTS))

    if extra_command:
        session.run(*extra_command, env=env)
    session.run(
        "uv",
        "run",
        "--no-dev",
        "--group",
        "test",
        *install_args,
        "pytest",
        *pytest_run_args,
        *session.posargs,
        "--cov-config=pyproject.toml",
        env=env,
    )


@nox.session(python=PYTHON_TEST_VERSIONS, reuse_venv=True, default=True)
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    _run_tests(session, run_rl_training=True)


@nox.session(python=PYTHON_TEST_VERSIONS, reuse_venv=True, venv_backend="uv", default=True)
def minimums(session: nox.Session) -> None:
    """Test the minimum versions of dependencies."""
    with preserve_lockfile():
        _run_tests(
            session,
            install_args=["--resolution=lowest-direct"],
            pytest_run_args=["-Wdefault"],
        )
        env = {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}
        session.run("uv", "tree", "--frozen", env=env)


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """Build the docs. Use "--non-interactive" to avoid serving. Pass "-b linkcheck" to check links."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", dest="builder", default="html", help="Build target (default: html)")
    args, posargs = parser.parse_known_args(session.posargs)

    serve = args.builder == "html" and session.interactive
    if serve:
        session.install("sphinx-autobuild")

    env = {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}
    shared_args = [
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        "docs",
        f"docs/_build/{args.builder}",
        *posargs,
    ]

    session.run(
        "uv",
        "run",
        "--no-dev",
        "--group",
        "docs",
        "--frozen",
        "sphinx-autobuild" if serve else "sphinx-build",
        *shared_args,
        env=env,
    )


if __name__ == "__main__":
    nox.main()
