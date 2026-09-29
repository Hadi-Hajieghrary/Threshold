"""Produce (and cache) one recorded mission from the event-rich Phase 1 cell, used by
the snapshot and cascade figures.

Cell i035_T0600_ks3 exactly as declared in records/v2/phase1/stochastic_declarations.json:
parallel formation, T0 = 600 N, weather intensity 0.35, k_h = 477 N m/rad, k_sigma = 3.0,
sway_limit = 0.349 rad, linear drag, recording cables, 300 s.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

from tether.campaign import fleet_run as FR
from tether.analysis.v2.paperfig.style import REPO

CACHE = REPO / "records" / "v2" / "figures" / "cascade_run.npz"

SPEC = dict(formation="parallel", pretension=600.0, heading_gain=477.0, weather_scale=0.35,
            k_sigma=3.0, sway_limit=0.349, drag_law="linear", cable_mode="recording",
            duration=300.0, log_state=True)


def simulate(seed: int):
    spec = FR.FleetRunSpec(**SPEC)
    run = FR.build_run(spec, seed)
    FR.run_to_end(run)
    log = run.fleet.cables.log
    n, m = log.count, log.state_count
    return dict(
        seed=seed,
        time=log.event_time[:n].copy(),
        elongation=log.elongation[:n].copy(),
        rate=log.rate[:n].copy(),
        alive=log.alive[:n].copy(),
        state_time=log.state_time[:m].copy(),
        state=log.state[:m].copy(),
        geometry_load=run.fleet.geometry.load_offsets.copy(),
        geometry_vessel=run.fleet.geometry.vessel_offsets.copy(),
        tensions0=run.fleet.operating.tensions.copy(),
    )


def tension(elong, rate, alive):
    """The plant's own unilateral Kelvin-Voigt law (tether/physics/fleet.py scalar_cable_step):
    k e + c edot, applied only while the chord is stretched (e > 0) and the cable is alive,
    then clamped at zero.  Constants come from the plant, not from this module."""
    from tether.physics import constants as K
    t = K.CABLE_STIFFNESS * elong + K.CABLE_DAMPING * rate
    return np.where(alive & (elong > 0.0) & (t > 0.0), t, 0.0)


def find_cascades(rec, within=3.0, min_peak=4500.0):
    """Re-engagements of peak >= min_peak followed by a slack onset on ANOTHER cable
    within `within` seconds.  Returns a list of (t_snap, parent_cable, peak, children)."""
    t = rec["time"]
    T = tension(rec["elongation"], rec["rate"], rec["alive"])
    slack = ~(rec["elongation"] > 0.0)
    out = []
    for i in range(T.shape[1]):
        # re-engagement = upcrossing of e = 0
        up = np.flatnonzero((~slack[:-1, i]) & (slack[1:, i]))  # taut -> slack (onset)
        re = np.flatnonzero((slack[:-1, i]) & (~slack[1:, i]))  # slack -> taut (re-engagement)
        for r in re:
            w = (t >= t[r]) & (t <= t[r] + 0.4)
            peak = float(T[w, i].max()) if w.any() else 0.0
            if peak < min_peak:
                continue
            kids = []
            for j in range(T.shape[1]):
                if j == i:
                    continue
                onsets = np.flatnonzero((~slack[:-1, j]) & (slack[1:, j]))
                hit = [t[o] for o in onsets if 0.0 < t[o] - t[r] <= within]
                if hit:
                    kids.append((j, float(min(hit))))
            if kids:
                out.append((float(t[r]), i, peak, kids))
    out.sort(key=lambda z: -z[2])
    return out


def main(seeds=(7101, 7102, 7103, 7104, 7105, 7106)):
    best = None
    for s in seeds:
        rec = simulate(s)
        casc = find_cascades(rec)
        n_multi = sum(1 for c in casc if len(c[3]) >= 2)
        print(f"seed {s}: {len(casc)} cascades, {n_multi} with >=2 children, "
              f"max peak {casc[0][2]/1e3:.1f} kN" if casc else f"seed {s}: none")
        if casc and (best is None or casc[0][2] > best[1][0][2]):
            best = (rec, casc)
    if best is None:
        raise SystemExit("no cascade found in the scanned seeds")
    rec, casc = best
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    ev = np.array([[c[0], c[1], c[2]] for c in casc])
    np.savez_compressed(CACHE, cascade_events=ev, **rec)
    print(f"\ncached seed {rec['seed']} -> {CACHE.relative_to(REPO)}")
    print(f"best cascade: t = {casc[0][0]:.2f} s, cable {casc[0][1]}, "
          f"peak {casc[0][2]/1e3:.2f} kN, children {casc[0][3]}")


if __name__ == "__main__":
    main()
