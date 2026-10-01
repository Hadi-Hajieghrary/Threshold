# Presentation clips

The nine simulation clips of the journal manuscript **Cable Severance in Cooperative Towing is a Fleet
Problem: Limits of Per-Line Prediction and of Thrust-Based Mitigation**. Each clip is a 1920 × 1080,
30 fps video. Every frame shows plant state, either from a recorded campaign run or from a replay of
one. Before drawing any frame, a replay asserts in code that it reproduces the recorded run
(configuration hash, re-engagement marks, peak tensions).

Each clip is available as a video in this folder and as a GIF rendering (880 px wide, 10 fps, full
length) with the same name in [`gifs/`](gifs/). The generator name is the clip's name in its script
under `tether/analysis/v2/present/` (or `tether/analysis/v3/present/` for `intervention`). The
generators write to `Presentation/clips/<generator name>.mp4`; the files here are those renders,
renamed.

| # | Video | GIF | Generator name | Length |
|---|---|---|---|---|
| 1 | [`The Operation.mp4`](The%20Operation.mp4) | [`The Operation.gif`](gifs/The%20Operation.gif) | `setting` | 70 s |
| 2 | [`Anatomy.mp4`](Anatomy.mp4) | [`Anatomy.gif`](gifs/Anatomy.gif) | `snap_cascade` | 106 s |
| 3 | [`Cascade_Statistics.mp4`](Cascade_Statistics.mp4) | [`Cascade_Statistics.gif`](gifs/Cascade_Statistics.gif) | `cascade_statistics` | 64 s |
| 4 | [`Intervention.mp4`](Intervention.mp4) | [`Intervention.gif`](gifs/Intervention.gif) | `intervention` (plan v3) | 103 s |
| 5 | [`Prediction.mp4`](Prediction.mp4) | [`Prediction.gif`](gifs/Prediction.gif) | `prediction` | 146 s |
| 6 | [`Mitigation_1.mp4`](Mitigation_1.mp4) | [`Mitigation_1.gif`](gifs/Mitigation_1.gif) | `easing` | 149 s |
| 7 | [`Mitigation_2.mp4`](Mitigation_2.mp4) | [`Mitigation_2.gif`](gifs/Mitigation_2.gif) | `catch` | 118 s |
| 8 | [`Slack_Criterion.mp4`](Slack_Criterion.mp4) | [`Slack_Criterion.gif`](gifs/Slack_Criterion.gif) | `slack_criterion` | 69 s |
| 9 | [`Pretension.mp4`](Pretension.mp4) | [`Pretension.gif`](gifs/Pretension.gif) | `pretension` | 81 s |

**Conventions used in every clip.** A taut cable is drawn solid, with a width that grows with its
tension (capped at 12 kN). A cable that carries no tension is drawn dashed red. The payload is the
plant's pentagon and the tugs are its 3 m hulls, drawn to scale. *T0* is the pretension, the steady
towing pull. A *snap* is the tension pulse produced when a slack cable is pulled taut again (a
*re-engagement*). In *recording* runs, severance is scored at a 4.5 kN stress threshold, but no cable
is cut.

The full account of each clip, including its selection rule and caveats, is in the
[root README](../README.md#the-simulations-in-motion).

---

## 1. `The Operation.mp4`: one squall-passage mission

One recorded squall-passage mission (seed 5008), played at four times real speed. Five tugs in fan
formation tow one payload on 12 m cables at T0 = 1 kN, through a thrust ramp, a 60° dogleg to port, a
squall from port, and a deceleration. In the squall the payload swings about 120° to starboard, drifts
89 m downwind, and the fan collapses. Cable 4 snaps taut at 13.4 kN at 58.8 s. Later, all five cables
go slack and snap back at up to 17.0 kN. The clip ends with the study's question: per-line practice
predicts a snap's peak from that line's own closing speed, but does that speed still belong to one
line when the lines share one payload?

## 2. `Anatomy.mp4`: anatomy of a snap and a cascade

One recorded cascade, slowed to 1/20 of real speed (parallel formation, T0 = 0.6 kN, seed 7105).
Cable 2 re-engages at 91.486 s and peaks at 13.5 kN, 22.5 times T0. Within 70 ms, cables 0, 4 and 3
go slack, and two of them then snap back. A closing slide gives the impulsive transmission estimate
(a neighbour's tension swings by up to about 0.44 of the peak). The slide labels this value derived,
not measured. The plan v3 campaign later measured 0.148, and the ACC 2027 paper explains why the
impulsive estimate overpredicts.

## 3. `Cascade_Statistics.mp4`: cascades against a chance baseline

A timeline of slack spells and re-engagements from one cascade-grid mission. A slack event is
highlighted when another cable re-engaged within the preceding 3 s. To build a chance baseline, the
re-engagement train is time-shifted, which keeps the event rates but removes causal timing. In the
mission shown, the shift lowers the highlighted share from 62% to about 15–18%. In all 18 scored cells,
the measured share exceeds the baseline's 97.5th percentile.

## 4. `Intervention.mp4`: the same moment, with and without the snap

One recorded snap (17.4 kN) is replayed twice from the same plant state: as recorded on the left, and
on the right with the snapping cable applying no force from re-engagement on. Within 3 s, 14 slack
onsets on other cables follow the real snap, and none occur without it. A second segment shows how
such counts form a branching matrix, whose spectral radius ρ indicates whether cascades die out or
grow, and plots ρ along a weather-intensity sweep. The re-integrator narrowly failed its
pre-registered validation, so the counterfactual counts are reported as untrusted.

## 5. `Prediction.mp4`: per-line forecast against a fleet rollout

The mission of clip 1, from 71 s. Every 0.1 s, two models forecast the probability of a dangerous snap
of cable 4 within 1 s. The per-line model (red) looks at cable 4 alone and forecasts certainty at six
ticks. The six-body fleet rollout (blue) forecasts zero at those ticks. When cables 1 and 2 snap first,
they jerk the payload, and cable 4's closing speed drops, so the six certain forecasts were false
alarms. Across all 40 missions, 24 of the per-line model's 25 high-confidence false alarms fell on
ticks where another cable had re-engaged within 1 s. The per-line model is not noisy but incomplete.

## 6. `Mitigation_1.mp4`: easing the fleet's thrust

Paired live missions with the same seed and weather; in these missions the simulator cuts any cable
that reaches 4.5 kN. The left arm has no supervisor. On the right, a supervisor eases each vessel's
thrust by up to 60% based on the fleet-wide hazard. The clip shows three seeds: both arms sever, only
the unsupervised arm severs, and only the supervised arm severs. Over 60 pairs, each arm loses a cable
in 25 missions (paired difference 0.00). Easing the fleet's thrust left severance unchanged.

## 7. `Mitigation_2.mp4`: the velocity-matching catch

A catch law brakes a slack tug so that its line comes taut slowly. It is tested as a counterfactual
from the deepest point of recorded dangerous excursions, in two variants: thrust-hold and
thrust-reverse. In the typical failure (seed 5020), braking lowers the closing speed, yet the peak
tension rises from 8.6 to about 10 kN. The one save (seed 5028) brings the cable taut at 0.89 of the
severing speed. Over 16 excursions the catch saves 1, against a pre-declared bar of 0.80.

## 8. `Slack_Criterion.mp4`: drag decides which gusts hold a line slack

Three scripted 10 s gusts on one vessel, with gust size λ = 0.9, 1.05 and 1.5 relative to the
predicted crossing λ = 1. At 0.9 the slack stays under 1 cm and soon ends. At 1.05 it deepens
throughout the gust, and the line then snaps taut at 14.9 kN. At 1.5 the formation closes. The closing
chart shows the fitted crossings averaging λ = 0.989, against a predicted 1.000; an inertial criterion
had predicted 1.055–1.077. The criterion involves one cable only, so per-line practice can adopt it
unchanged.

## 9. `Pretension.mp4`: one knob, two requirements

The same seed and weather at T0 = 0.6 kN (left) and 1.0 kN (right). The left run contains the cascade
of clip 2, and the right run never goes slack. Over 20 runs each, raising the pretension narrows the
chord-angle spread from 35.9° to 20.9° and cuts re-engagements from 510 to 4 (−99.2%). No setting on
the grid meets the 15° shape band; the best misses it by a factor of 2.39. More pretension steadies
the shape, but it suppresses the slack events that a design law needs. The two cells also differ in
heading gain, so the comparison is not pretension alone.
