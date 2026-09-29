"""Shared campaign plumbing: deterministic records, manifests, and the process pool."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import zipfile
from multiprocessing import get_context
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT / "records"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DEFAULT_WORKERS = int(os.environ.get("TETHER_WORKERS", "16"))


def json_bytes(value: object) -> bytes:
    def default(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    return (json.dumps(value, indent=2, sort_keys=True, default=default, allow_nan=True) + "\n").encode("ascii")


def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Serialize an NPZ with fixed member order, metadata, and timestamps."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            array_buffer = io.BytesIO()
            np.lib.format.write_array(array_buffer, np.asarray(arrays[name]), allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            archive.writestr(member, array_buffer.getvalue())
    return buffer.getvalue()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def config_sha256(configuration: object) -> str:
    return sha256_bytes(json.dumps(configuration, sort_keys=True, separators=(",", ":"), default=str).encode("ascii"))


def source_state() -> dict[str, object]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "status": "unavailable"}
    return {"revision": revision, "status": "dirty" if status.strip() else "clean"}


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256_bytes(payload)


def _worker_init() -> None:
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"


def run_pool(
    function: Callable,
    jobs: Sequence,
    workers: int = DEFAULT_WORKERS,
    progress: Callable[[int, int], None] | None = None,
) -> list:
    """Run ``function`` over ``jobs`` in worker processes, preserving job order.

    One job is one (cell, seed): results never depend on scheduling.  Jobs are
    submitted longest-first when they carry a ``cost`` attribute.
    """
    if workers <= 1 or len(jobs) <= 1:
        return [function(job) for job in jobs]
    order = sorted(range(len(jobs)), key=lambda i: -float(getattr(jobs[i], "cost", 1.0)))
    results: list = [None] * len(jobs)
    with get_context("spawn").Pool(processes=min(workers, len(jobs)), initializer=_worker_init, maxtasksperchild=4) as pool:
        for done, (index, result) in enumerate(
            pool.imap_unordered(_indexed_call, [(i, function, jobs[i]) for i in order], chunksize=1), start=1
        ):
            results[index] = result
            if progress is not None:
                progress(done, len(jobs))
    return results


def _indexed_call(payload):
    index, function, job = payload
    return index, function(job)


def concatenate_tables(tables: Iterable[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    tables = [table for table in tables if table]
    if not tables:
        return {}
    keys = tables[0].keys()
    return {key: np.concatenate([np.asarray(table[key]) for table in tables]) for key in keys}


def verdict_line(passed: bool | None) -> str:
    if passed is None:
        return "UNDER-POWERED"
    return "PASS" if passed else "FAIL"
