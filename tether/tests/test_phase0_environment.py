"""Pinned Phase 0 environment acceptance test."""

import importlib.metadata
from pathlib import Path

import pydrake


def test_p0_t1_exact_metadata_versions():
    expected = {
        "drake": "1.51.1",
        "numpy": "2.2.6",
        "scipy": "1.15.3",
        "matplotlib": "3.10.9",
        "pytest": "8.4.2",
    }
    lock_path = Path(__file__).parents[2] / "requirements.lock.txt"
    locked = dict(
        line.split("==", maxsplit=1)
        for line in lock_path.read_text(encoding="ascii").splitlines()
        if line
    )

    assert not hasattr(pydrake, "__version__")
    assert locked == expected
    assert {package: importlib.metadata.version(package) for package in expected} == expected