"""Generic stationary (cell, seed) jobs shared by Phases 2-4."""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from tether.campaign.common import run_pool
from tether.campaign.fleet_run import FleetRunSpec, build_run, run_to_end
from tether.campaign.summaries import summarize_run


@dataclass(frozen=True)
class StationaryJob:
    cell: str
    spec: FleetRunSpec
    seed: int
    pilot: bool = False

    @property
    def cost(self) -> float:
        return self.spec.duration + self.spec.warmup


def run_stationary_job(job: StationaryJob) -> dict:
    run = run_to_end(build_run(job.spec, job.seed))
    summary = summarize_run(run, job.spec.warmup, job.spec.pretension)
    summary["job"] = {"cell": job.cell, "seed": job.seed, "pilot": job.pilot, "spec": asdict(job.spec)}
    return summary


def compute_jobs(jobs: list[StationaryJob], cache_path: Path, workers: int, label: str) -> list[dict]:
    """Run the jobs (resuming from a partial cache) and cache every summary."""
    done: dict[tuple[str, int], dict] = {}
    partial = cache_path.with_suffix(".partial.pkl")
    if partial.exists():
        with partial.open("rb") as handle:
            done = pickle.load(handle)
    pending = [job for job in jobs if (job.cell, job.seed) not in done]
    batch = max(workers * 2, 1)
    for start in range(0, len(pending), batch):
        chunk = pending[start : start + batch]
        results = run_pool(run_stationary_job, chunk, workers)
        for job, summary in zip(chunk, results):
            done[(job.cell, job.seed)] = summary
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with partial.open("wb") as handle:
            pickle.dump(done, handle)
        print(f"{label}: {len(done)}/{len(jobs)} jobs", flush=True)
    summaries = [done[(job.cell, job.seed)] for job in jobs]
    with cache_path.open("wb") as handle:
        pickle.dump({"jobs": jobs, "summaries": summaries}, handle)
    partial.unlink(missing_ok=True)
    return summaries


def load_cache(cache_path: Path) -> tuple[list[StationaryJob], list[dict]]:
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload["jobs"], payload["summaries"]


def with_seed(spec: FleetRunSpec, **changes) -> FleetRunSpec:
    return replace(spec, **changes)
