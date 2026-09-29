"""Assemble the presentation: slides + clips -> Presentation/presentation.mp4 (and a 720p copy),
with chapter markers, a timed speaker script (SCRIPT.md) and the provenance of every on-screen
number (PROVENANCE.md), both compiled from the manifests the clips and slides wrote.

Run after every clip exists:  python3 -m tether.analysis.v2.present.assemble
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tether.analysis.v2.present import common as C

# (kind, name, chapter title) in running order
ORDER = [
    ("slide", "s01_title", "Title"),
    ("slide", "s01b_problem", "The problem"),
    ("slide", "s02_why", "Why it matters"),
    ("slide", "s02c_contrib", "What this paper contributes"),
    ("slide", "s02d_contrib", "What this paper contributes (continued)"),
    ("slide", "s02b_legend", "How to read the movies"),
    ("clip", "setting", "The operation: a squall-passage mission"),
    ("slide", "s03_perline", "How snaps are treated today"),
    ("slide", "p1_mechanism", "Part 1 - Does a snap on one line affect the others?"),
    ("clip", "snap_cascade", "Anatomy of a snap and a cascade"),
    ("slide", "p1b_evidence", "Part 1 - Is it more than a coincidence?"),
    ("clip", "cascade_statistics", "Cascades against a chance baseline"),
    # plan v3 (cascade criticality): present once the intervention clip's manifest exists (make v3-clip)
    *([("slide", "p1c_intervention", "Part 1 - Does removing the snap remove the follow-on slack?"),
       ("clip", "intervention", "The same moment, with and without the snap"),
       ("slide", "s04b_criticality", "The cascade as a branching process")]
      if (C.MANIFESTS / "intervention.json").exists() else []),
    ("slide", "s04_roadmap", "What the mechanism implies"),
    ("slide", "p2_prediction", "Part 2 - Can one line's danger be forecast from that line alone?"),
    ("clip", "prediction", "Per-line hazard vs fleet rollout"),
    ("slide", "s05a_prediction_result", "Prediction: the result"),
    ("slide", "s05_prediction", "Prediction: what it does and does not show"),
    ("slide", "p3_mitigation", "Part 3 - Can thrust control prevent the snaps?"),
    ("clip", "easing", "Easing the fleet's thrust: paired missions"),
    ("slide", "p3b_catch", "Part 3 - Can the slack tug catch its line gently?"),
    ("clip", "catch", "The velocity-matching catch"),
    ("slide", "s06_mitigation", "Mitigation: the result"),
    ("slide", "p4_design", "Part 4 - Can a formula tell designers how often lines will break?"),
    ("clip", "slack_criterion", "The drag-conjugate slack criterion"),
    ("slide", "p4b_pretension", "Part 4 - The pretension trade"),
    ("clip", "pretension", "Shape against slack events"),
    ("slide", "s07_design", "Design time: the result"),
    ("slide", "s08_versus", "Without and with the fleet view"),
    ("slide", "s09_limits", "Limits"),
    ("slide", "s10_end", "End"),
]


def _path(kind: str, name: str) -> Path:
    return C.CLIPS / ("slides" if kind == "slide" else "") / f"{name}.mp4"


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_packets",
                          "-show_entries", "format=duration", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    j = json.loads(out)
    st, fmt = j["streams"][0], j["format"]
    return {"codec": st["codec_name"], "w": st["width"], "h": st["height"], "pix": st["pix_fmt"],
            "fps": st["r_frame_rate"], "frames": int(st["nb_read_packets"]), "duration": float(fmt["duration"])}


def _fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def main() -> None:
    slides = {s["name"]: s for s in json.loads((C.MANIFESTS / "slides_segments.json").read_text())}
    slide_man = json.loads((C.MANIFESTS / "slides.json").read_text())
    segs, t = [], 0.0
    for kind, name, chapter in ORDER:
        p = _path(kind, name)
        assert p.exists(), f"missing segment {p}"
        info = probe(p)
        assert (info["codec"], info["w"], info["h"], info["pix"], info["fps"]) == ("h264", C.W, C.H, "yuv420p", f"{C.FPS}/1"), (name, info)
        man = slide_man if kind == "slide" else json.loads((C.MANIFESTS / f"{name}.json").read_text())
        if kind == "clip":
            assert man["frames"] == info["frames"], f"{name}: manifest frames {man['frames']} != video {info['frames']}"
            caps = man.get("captions", [])
        else:
            caps = [c if isinstance(c, dict) else {"t": None, "text": c} for c in slides[name]["captions"]]
        segs.append({"kind": kind, "name": name, "chapter": chapter, "path": p, "start": t,
                     "duration": info["frames"] / C.FPS, "manifest": man, "captions": caps})
        t += info["frames"] / C.FPS
    total = t
    print(f"{len(segs)} segments, total {total:.1f} s ({total / 60:.2f} min)")
    assert total >= 8 * 60, "the presentation must run at least 8 minutes"

    # concatenate (re-encode once so every boundary is clean), then add chapters
    lst = C.OUT / "concat.txt"
    lst.write_text("".join(f"file '{s['path']}'\n" for s in segs))
    raw = C.OUT / "presentation_nochapters.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c:v", "libx264", "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p",
                    "-r", str(C.FPS), str(raw)], check=True)
    meta = C.OUT / "chapters.txt"
    body = [";FFMETADATA1", "title=Cable Severance in Cooperative Towing is a Fleet Problem"]
    for s in segs:
        body += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(round(s['start'] * 1000))}",
                 f"END={int(round((s['start'] + s['duration']) * 1000))}", f"title={s['chapter']}"]
    meta.write_text("\n".join(body) + "\n")
    final = C.OUT / "presentation.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-i", str(meta), "-map_metadata", "1",
                    "-map_chapters", "1", "-c", "copy", "-movflags", "+faststart", str(final)], check=True)
    raw.unlink()
    small = C.OUT / "presentation_720p.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(final), "-vf", "scale=1280:720", "-c:v", "libx264",
                    "-crf", "24", "-preset", "medium", "-pix_fmt", "yuv420p", "-map_metadata", "0",
                    "-map_chapters", "0", "-movflags", "+faststart", str(small)], check=True)
    fin = probe(final)
    print(f"wrote {final.relative_to(C.REPO)}: {fin['duration']:.1f} s, {fin['frames']} frames; "
          f"{small.relative_to(C.REPO)}")

    # speaker script
    lines = ["# Speaker script and running order", "",
             f"Total running time **{_fmt(total)}** ({total / 60:.1f} min) - `Presentation/presentation.mp4` "
             "(1920x1080, 30 fps; chapters embedded). The burned-in captions are the narration; this script "
             "repeats them with their times so the talk can also be given live.", ""]
    for s in segs:
        m = s["manifest"]
        lines += [f"## {_fmt(s['start'])} - {s['chapter']}  ({s['duration']:.0f} s, {s['kind']} `{s['name']}`)", ""]
        if s["kind"] == "clip":
            lines += [m.get("story", ""), ""]
            if m.get("selection"):
                lines += [f"*Selection:* {m['selection']}", ""]
        for c in s["captions"]:
            stamp = _fmt(s["start"] + c["t"]) if c["t"] is not None else "·"
            lines.append(f"- `{stamp}` {c['text']}")
        if s["kind"] == "clip" and m.get("caveats"):
            lines += ["", "*Caveats:* " + " · ".join(m["caveats"])]
        lines.append("")
    (C.OUT / "SCRIPT.md").write_text("\n".join(lines))

    # provenance
    pv = ["# Provenance of every on-screen number", "",
          "Compiled from `Presentation/manifests/*.json`. Each clip asserts, before drawing, that its "
          "replays reproduce the campaign records (the checks listed per clip).", ""]
    seen = set()
    for s in segs:
        key = "slides" if s["kind"] == "slide" else s["name"]
        if key in seen:
            continue
        seen.add(key)
        m = s["manifest"]
        pv += [f"## `{key}` - {m.get('title', '')}", ""]
        if m.get("checks"):
            pv += ["Reproduction and consistency checks (all passed):", ""]
            pv += [f"- {c['check']}" + (f" - {c['detail']}" if c.get("detail") else "") for c in m["checks"]]
            pv.append("")
        pv += ["| On screen | Value | Unit | Source | Note |", "|---|---|---|---|---|"]
        for v in m.get("values", []):
            val = json.dumps(v["value"], default=float)
            pv.append(f"| {v['label']} | `{val[:120]}` | {v.get('unit', '')} | {v.get('source', '')} | {v.get('note', '')} |")
        pv += ["", "Sources (sha256):", ""] + [f"- `{k}` `{h[:16]}…`" for k, h in m.get("sources", {}).items()] + [""]
    (C.OUT / "PROVENANCE.md").write_text("\n".join(pv))
    (C.OUT / "running_order.json").write_text(json.dumps(
        [{k: (str(v.relative_to(C.REPO)) if isinstance(v, Path) else v) for k, v in s.items() if k != "manifest"} for s in segs],
        indent=1, default=float))
    print("wrote Presentation/SCRIPT.md, PROVENANCE.md, running_order.json")


if __name__ == "__main__":
    main()
