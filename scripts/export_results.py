"""Export numeric LIBERO results without model weights, media, or replay buffers.

Uses only the Python standard library. Source files are never modified.
"""

import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path


NAMES = {
    "config.json", "manifest.json", "planned_suite.json", "planned_runs.json",
    "experiment.json", "suite.json", "status.json", "summary.json",
    "reward_provenance.json", "provenance.json", "flags.json",
    "evaluation.csv", "eval.csv", "eval_sde.csv", "curves.csv",
    "online_agent.csv", "offline_agent.csv", "metrics.jsonl",
    "episodes.jsonl", "episodes.json", "diagnostics.jsonl",
    "statistics.json", "seed_statistics.json", "sparse_auc_statistics.json",
    "curves.json", "iteration_curves.json", "report_data.json",
}
SKIP = {"videos", "frames", "checkpoints", "ckpts", "replay_buffer",
        "replay_buffers", "dataset", "datasets", "cache", "__pycache__"}
PREFIXES = ("pi0", "reward_audit_", "sac_task28_", "dsrl_")
EXCLUDE_ROOTS = {"pi0_cache", "pi0_checkpoints", "dsrl_official", "dsrl_official_smoke"}
EXTRA_ROOTS = {"experiments", "experiment_workers", "experiment_specs", "video_audits", "ogpo_libero"}


def walk(folder, source, ancestors=()):
    real = folder.resolve()
    if not real.is_relative_to(source) or real in ancestors:
        return
    for path in sorted(folder.iterdir()):
        if path.name.startswith("."):
            continue
        if path.is_dir():
            if path.name.lower() not in SKIP:
                yield from walk(path, source, (*ancestors, real))
        elif path.is_file() and path.resolve().is_relative_to(source):
            if path.name in NAMES or (folder.name == "experiment_specs" and path.suffix == ".json"):
                yield path


def read_snapshot(path):
    # Bound reads to the size at opening; writers may append during export.
    with path.open("rb") as stream:
        import os
        raw = stream.read(os.fstat(stream.fileno()).st_size)
    if path.suffix == ".jsonl":
        rows = raw.splitlines(keepends=True)
        kept = []
        for i, row in enumerate(rows):
            try:
                json.loads(row)
            except (ValueError, UnicodeDecodeError):
                if i == len(rows) - 1 and not row.endswith(b"\n"):
                    break
                raise
            kept.append(row.rstrip(b"\r\n") + b"\n")
        return b"".join(kept)
    if path.suffix == ".csv" and raw and not raw.endswith(b"\n"):
        # Training CSV writers append newline-terminated records.
        raw = raw[:raw.rfind(b"\n") + 1]
    if path.suffix == ".json":
        json.loads(raw)
    return raw


def export(source, output):
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise ValueError(f"Use a new snapshot directory; refusing to overwrite {output}")
    if output.is_relative_to(source):
        raise ValueError("Keep exported snapshots outside the source data tree")
    output.mkdir(parents=True)
    roots = [p for p in sorted(source.iterdir()) if p.is_dir()
             and p.name not in EXCLUDE_ROOTS
             and (p.name.startswith(PREFIXES) or p.name in EXTRA_ROOTS)]
    roots.sort(key=lambda p: (0 if p.name == "experiments" else
                              1 if (p / "manifest.json").exists() else
                              3 if p.name == "experiment_workers" else 2, p.name))
    files, warnings, canonical = [], [], {}
    for root in roots:
        for path in walk(root, source):
            relative = path.relative_to(source)
            try:
                raw = read_snapshot(path)
            except (ValueError, OSError) as error:
                warnings.append({"path": str(relative), "error": str(error)})
                continue
            target = output / "data" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            real = str(path.resolve())
            entry = {"path": str(relative), "bytes": len(raw),
                     "sha256": hashlib.sha256(raw).hexdigest()}
            if real in canonical:
                entry["alias_of"] = canonical[real]
            else:
                canonical[real] = str(relative)
            files.append(entry)

    # A portable table for the SAC/DSRL curves; absolute config paths are provenance only.
    points, runs, frozen = [], [], []
    for entry in files:
        relative = Path(entry["path"])
        if "alias_of" in entry:
            continue
        folder = output / "data" / relative.parent
        cfg_path = folder / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        if relative.name == "summary.json":
            summary = json.loads((folder / relative.name).read_text())
            if isinstance(summary, dict) and "success_rate" in summary:
                frozen.append({"path": str(relative), "config": cfg, "summary": summary})
        if relative.name != "evaluation.csv":
            continue
        count = cfg.get("num_envs", 1)
        metadata = {
            "run": str(relative.parent), "suite": cfg.get("suite", ""),
            "task": cfg.get("task", ""), "task_id": cfg.get("task_id", ""),
            "reward": cfg.get("reward", ""), "seed": cfg.get("seed", ""),
            "base_model": cfg.get("base_model") or cfg.get("policy_config_name")
            or ("pi0" if cfg.get("pi0_checkpoint") else "scratch"),
            "num_envs": count,
        }
        rows = list(csv.DictReader(io.StringIO((folder / relative.name).read_text())))
        for row in rows:
            try:
                steps = float(row["env_steps"])
                successes, episodes = int(row["successes"]), int(row["episodes"])
                rate = float(row["success_rate"])
                if episodes <= 0 or not 0 <= successes <= episodes or abs(rate - successes / episodes) > 1e-7:
                    raise ValueError("Inconsistent success counts")
                points.append({**metadata, "physical_steps_total": steps,
                               "physical_steps_per_env": steps / count,
                               "successes": successes, "eval_episodes": episodes,
                               "success_rate": rate})
            except (ValueError, KeyError, TypeError) as error:
                warnings.append({"path": str(relative), "error": str(error), "row": row})
        runs.append({**metadata, "evaluations": len(rows), "config": str(relative.parent / "config.json")})
    if points:
        with (output / "evaluation_points.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(points[0]))
            writer.writeheader()
            writer.writerows(points)
    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    write("runs.json", runs)
    write("frozen_evaluations.json", frozen)
    write("index.json", {
        "schema_version": 1, "exported_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source), "roots": [p.name for p in roots], "files": files,
        "warnings": warnings, "bytes": sum(e["bytes"] for e in files),
        "note": "Read-only, file-by-file snapshot; active runs are not transactionally frozen. "
                "Aliases are materialized but deduplicated in evaluation_points.csv. "
                "OGPO CSVs retain native step units and are not merged into that table.",
    })
    print(json.dumps({"output": str(output), "files": len(files), "runs": len(runs),
                      "evaluations": len(points), "frozen_summaries": len(frozen),
                      "bytes": sum(e["bytes"] for e in files), "warnings": len(warnings)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("local_results"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.out)
