# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Nox sessions."""

from __future__ import annotations

import argparse
import contextlib
import os
import pathlib
import shutil
import tempfile
from typing import TYPE_CHECKING

import nox

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv"

nox.options.sessions = ["lint", "tests", "minimums"]


# TODO(denialhaag): Add 3.14 when all dependencies support it
#   https://github.com/munich-quantum-toolkit/predictor/issues/420
PYTHON_ALL_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]

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


@nox.session(reuse_venv=True)
def lint(session: nox.Session) -> None:
    """Run the linter."""
    if shutil.which("pre-commit") is None:
        session.install("pre-commit")

    session.run("pre-commit", "run", "--all-files", *session.posargs, external=True)


def _cleanup(session: nox.Session) -> None:
    """Remove this session's virtualenv to save disk space in CI."""
    venv_dir = session.virtualenv.location
    if venv_dir and pathlib.Path(venv_dir).exists():
        shutil.rmtree(venv_dir, ignore_errors=True)
        session.log(f"Cleaned up {venv_dir}")
    shutil.rmtree(pathlib.Path("~/.cache").expanduser(), ignore_errors=True)
    # Clean GitHub Actions temp cache
    gha_temp = pathlib.Path("/home/runner/work/_temp/setup-uv-cache")
    if gha_temp.exists():
        shutil.rmtree(gha_temp, ignore_errors=True)
        session.log(f"Cleaned GitHub Actions uv temp cache at {gha_temp}")


def _run_tests(
    session: nox.Session,
    *,
    install_args: Sequence[str] = (),
    extra_command: Sequence[str] = (),
    pytest_run_args: Sequence[str] = (),
) -> None:
    env = {"UV_PROJECT_ENVIRONMENT": session.virtualenv.location}

    if extra_command:
        session.run(*extra_command, env=env)
    if "--cov" in session.posargs:
        # Try to use the lighter-weight `sys.monitoring` coverage core available in Python 3.12+ to speed up coverage collection.
        env["COVERAGE_CORE"] = "sysmon"
    session.run(
        "uv",
        "run",
        "--no-dev",
        "--no-cache",
        "--group",
        "test",
        *install_args,
        "pytest",
        *pytest_run_args,
        *session.posargs,
        "--cov-config=pyproject.toml",
        env=env,
    )


@nox.session(reuse_venv=not os.environ.get("CI"), python=PYTHON_ALL_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    _run_tests(session)


@nox.session(reuse_venv=True, venv_backend="uv", python=PYTHON_ALL_VERSIONS)
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
