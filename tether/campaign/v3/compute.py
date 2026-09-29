"""Plan v3 stage runner: build, run and reduce every (cell, seed) job, resumably, in a process pool.

    python -m tether.campaign.v3.compute stage-a --workers 16     # WP1 + WP2 runs (16 cells)
    python -m tether.campaign.v3.compute stage-b --workers 16     # WP3 sweep; needs gate_wp2 OPEN
    python -m tether.campaign.v3.compute stage-b-ext --workers 16 # needs records/v3/b_ext_authorised.json

Every stage refuses to run unless the declarations verify (tether.campaign.v3.campaign.verify_declared).
Caches hold the reduced dicts only (records/v3/cache/v3_stage<X>.pkl, resumable through
.partial.pkl), never the 1 ms logs.  A probe (``replay <cell> <seed>``) re-runs one job and
reports whether its marks reproduce the cached ones bit-exactly.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time as wallclock
from pathlib import Path

import numpy as np

from tether.campaign.common import DEFAULT_WORKERS, run_pool
from tether.campaign.v3 import campaign, cells
from tether.campaign.v3.intervention import InterventionSpec

CACHE_DIR = campaign.RECORD_DIR / "cache"
STAGE_CACHE = {"A": CACHE_DIR / "v3_stageA.pkl", "B": CACHE_DIR / "v3_stageB.pkl", "B_ext": CACHE_DIR / "v3_stageB_ext.pkl"}
GATE_WP2 = campaign.RECORD_DIR / "gate_wp2.json"
B_EXT_AUTHORISATION = campaign.RECORD_DIR / "b_ext_authorised.json"
INTERVENTION = InterventionSpec()


def run_v3_job(job: cells.V3Job) -> dict:
    from tether.campaign.fleet_run import build_run, run_to_end
    from tether.campaign.v3.reducer import reduce_run

    started = wallclock.perf_counter()
    run = run_to_end(build_run(job.spec, job.seed))
    run.wall_seconds = wallclock.perf_counter() - started
    out = reduce_run(run, job, INTERVENTION)
    out["meta"]["wall_seconds_total"] = wallclock.perf_counter() - started
    return out


def compute_stage(stage: str, workers: int = DEFAULT_WORKERS, jobs: list | None = None) -> list[dict]:
    campaign.verify_declared()
    if stage == "B" and not gate_open(GATE_WP2):
        raise SystemExit("stage B (the WP3 sweep) needs records/v3/gate_wp2.json with verdict OPEN")
    if stage == "B_ext" and not B_EXT_AUTHORISATION.exists():
        raise SystemExit("stage B_ext needs records/v3/b_ext_authorised.json, written by the WP3 analysis under the declared rule")
    jobs = cells.all_jobs(stage) if jobs is None else jobs
    cache = STAGE_CACHE[stage]
    partial = cache.with_suffix(".partial.pkl")
    done: dict[tuple[str, int], dict] = {}
    if partial.exists():
        with partial.open("rb") as handle:
            done = pickle.load(handle)
    pending = [job for job in jobs if (job.cell, job.seed) not in done]
    batch = max(workers * 2, 1)
    started = wallclock.perf_counter()
    for start in range(0, len(pending), batch):
        chunk = pending[start : start + batch]
        results = run_pool(run_v3_job, chunk, workers)
        for job, out in zip(chunk, results):
            done[(job.cell, job.seed)] = out
        cache.parent.mkdir(parents=True, exist_ok=True)
        with partial.open("wb") as handle:
            pickle.dump(done, handle)
        elapsed = wallclock.perf_counter() - started
        print(f"v3 stage {stage}: {len(done)}/{len(jobs)} jobs, {elapsed / 60:.1f} min", flush=True)
    summaries = [done[(job.cell, job.seed)] for job in jobs]
    light = [{"cell": j.cell, "seed": j.seed, "pilot": j.pilot, "stage": j.stage, "v1_cell": j.v1_cell, "config_hash": j.spec.config_hash()} for j in jobs]
    with cache.open("wb") as handle:
        pickle.dump({"jobs": light, "summaries": summaries, "declarations_sha256": campaign.sha256_file(campaign.DECLARATIONS_PATH)}, handle)
    partial.unlink(missing_ok=True)
    return summaries


def load_stage(stage: str) -> tuple[list[dict], list[dict]]:
    with STAGE_CACHE[stage].open("rb") as handle:
        payload = pickle.load(handle)
    return payload["jobs"], payload["summaries"]


def gate_open(path: Path) -> bool:
    if not path.exists():
        return False
    return json.loads(path.read_text()).get("verdict") == "OPEN"


def replay(cell: str, seed: int) -> dict:
    """Re-run one cached job and compare its marks bit-exactly with the cache."""
    for stage in cells.STAGES:
        if not STAGE_CACHE[stage].exists():
            continue
        jobs, summaries = load_stage(stage)
        for light, summary in zip(jobs, summaries):
            if light["cell"] == cell and int(light["seed"]) == int(seed):
                job = next(j for j in cells.all_jobs(stage, expected=False) if j.cell == cell and j.seed == seed)
                fresh = run_v3_job(job)
                same = (np.array_equal(fresh["marks"]["t_up"], summary["marks"]["t_up"])
                        and np.array_equal(fresh["marks"]["T_peak"], summary["marks"]["T_peak"]))
                out = {"cell": cell, "seed": seed, "stage": stage, "marks_exact": bool(same), "n_marks": int(np.asarray(fresh["marks"]["t_up"]).size)}
                print(json.dumps(out))
                return out
    raise SystemExit(f"{cell} seed {seed} is in no v3 cache")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["stage-a", "stage-b", "stage-b-ext", "replay"])
    parser.add_argument("args", nargs="*")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args(argv)
    if args.command == "replay":
        replay(args.args[0], int(args.args[1]))
        return
    stage = {"stage-a": "A", "stage-b": "B", "stage-b-ext": "B_ext"}[args.command]
    summaries = compute_stage(stage, args.workers)
    wall = sum(s["meta"].get("wall_seconds_total", 0.0) for s in summaries)
    print(f"v3 stage {stage} done: {len(summaries)} runs, {wall / 3600:.1f} core-hours of run+reduce")


if __name__ == "__main__":
    main()
