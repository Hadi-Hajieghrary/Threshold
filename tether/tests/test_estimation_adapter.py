"""The online estimator adapter reproduces the offline replay exactly."""

import numpy as np

from tether.campaign.fleet_run import FleetRunSpec, build_run, run_to_end
from tether.estimation.adapter import EstimatorAdapter
from tether.estimation.replay import estimator_geometry, replay_bundle, run_arms
from tether.estimation.truth_isolation import lint_diagram


def _spec():
    return FleetRunSpec(formation="fan", arc_half_angle=0.55, pretension=1000.0, weather_scale=1.0, weather_direction="common",
                        weather_distribution="student_t3", duration=12.0, warmup=0.0, log_state=True)


def test_online_adapter_matches_offline_replay():
    spec = _spec()
    holder = {}

    def attach(builder, fleet, sensors, controller):
        geometry = estimator_geometry(fleet.geometry)
        from tether.physics.fleet import equilibrium_state

        state = equilibrium_state(fleet.geometry, fleet.operating)
        poses = state[: 3 * (fleet.geometry.vessel_count + 1)].reshape(-1, 3)
        adapter = builder.AddSystem(EstimatorAdapter("P", geometry, poses, 0.4, 7, np.full(5, 0.3), compute_hazard=False))
        adapter.set_name("estimator_adapter_P")
        n = fleet.geometry.vessel_count
        for i, suite in enumerate(sensors):
            builder.Connect(suite.GetOutputPort("odometry"), adapter.get_input_port(i))
            builder.Connect(suite.GetOutputPort("cable_bearing"), adapter.get_input_port(n + i))
            builder.Connect(suite.GetOutputPort("tension"), adapter.get_input_port(2 * n + i))
        builder.Connect(sensors[0].GetOutputPort("beacon"), adapter.get_input_port(3 * n))
        holder["adapter"] = adapter
        return {"adapter": adapter}

    run = build_run(spec, 7, pre_build=attach)
    assert lint_diagram(run.fleet.diagram) == []
    run_to_end(run)
    online = holder["adapter"].estimator.output()
    log, _ = replay_bundle(run)
    offline = run_arms(log, arms=("P",), tau=0.4, master_seed=7)["P"]
    count = min(online.time.size, offline.time.size)
    assert count > 100
    assert np.array_equal(online.time[:count], offline.time[:count])
    assert np.allclose(online.load_mean[:count], offline.load_mean[:count], rtol=0.0, atol=1.0e-9, equal_nan=True)
    assert np.array_equal(online.slack[:count], offline.slack[:count])
