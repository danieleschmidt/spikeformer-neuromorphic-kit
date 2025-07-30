"""Nox configuration for SpikeFormer Neuromorphic Kit.

Nox is a command-line tool that automates testing in multiple Python environments.
This configuration defines sessions for testing, linting, and other development tasks.
"""

import nox

# Python versions to test against
PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12"]
DEFAULT_PYTHON = "3.10"

# Package locations
PACKAGE_NAME = "spikeformer"
TEST_PATHS = ["tests/", "spikeformer/"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session):
    """Run unit and integration tests."""
    session.install("-e", ".[dev]")
    session.run(
        "pytest",
        "tests/unit/",
        "tests/integration/",
        "--cov=spikeformer",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "--cov-fail-under=80",
        "-v",
        *session.posargs,
    )


@nox.session(python=DEFAULT_PYTHON)
def hardware_tests(session):
    """Run hardware-specific tests (requires hardware access)."""
    session.install("-e", ".[dev,hardware]")
    session.run(
        "pytest",
        "tests/hardware/",
        "--hardware",
        "-v",
        *session.posargs,
    )


@nox.session(python=DEFAULT_PYTHON)
def lint(session):
    """Run linting checks."""
    session.install("-e", ".[dev]")
    session.run("ruff", "check", "spikeformer/", "tests/")
    session.run("black", "--check", "spikeformer/", "tests/")
    session.run("isort", "--check-only", "spikeformer/", "tests/")


@nox.session(python=DEFAULT_PYTHON)
def format(session):
    """Format code using black and isort."""
    session.install("-e", ".[dev]")
    session.run("black", "spikeformer/", "tests/")
    session.run("isort", "spikeformer/", "tests/")


@nox.session(python=DEFAULT_PYTHON)
def typecheck(session):
    """Run type checking with mypy."""
    session.install("-e", ".[dev]")
    session.run("mypy", "spikeformer/")


@nox.session(python=DEFAULT_PYTHON)
def security(session):
    """Run security checks."""
    session.install("-e", ".[dev]")
    session.run("bandit", "-r", "spikeformer/", "-f", "json", "-o", "bandit-report.json")
    session.run("safety", "check")
    session.run("pip-audit")


@nox.session(python=DEFAULT_PYTHON)
def docs(session):
    """Build documentation."""
    session.install("-e", ".[dev]")
    session.run("sphinx-build", "-b", "html", "docs/", "docs/_build/")


@nox.session(python=DEFAULT_PYTHON)
def benchmark(session):
    """Run performance benchmarks."""
    session.install("-e", ".[dev]")
    session.run("pytest", "tests/performance/", "--benchmark-only", "-v")


@nox.session(python=DEFAULT_PYTHON)
def coverage(session):
    """Generate coverage report."""
    session.install("-e", ".[dev]")
    session.run(
        "pytest",
        "tests/",
        "--cov=spikeformer",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-report=xml",
    )


@nox.session(python=DEFAULT_PYTHON)
def pre_commit(session):
    """Run pre-commit hooks."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files")


@nox.session(python=DEFAULT_PYTHON, venv_backend="conda")
def conda_tests(session):
    """Run tests in a conda environment (for conda compatibility)."""
    session.conda_install("--channel", "conda-forge", "pytest", "pytest-cov")
    session.install("-e", ".")
    session.run("pytest", "tests/unit/", "-v")


@nox.session(python=DEFAULT_PYTHON)
def build(session):
    """Build the package."""
    session.install("build", "twine")
    session.run("python", "-m", "build")
    session.run("twine", "check", "dist/*")