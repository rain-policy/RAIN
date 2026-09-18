"""Run the released RAIN policy on one or all ManiSkill tasks."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_tasks():
    return json.loads((ROOT / "manifests/tasks.json").read_text(encoding="utf-8"))


def verify(task):
    files = {}
    for role, name in task["files"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Model path leaves repository")
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}; run python tools/download_models.py"
            )
        if digest(path) != task["hashes"][role]:
            raise ValueError(f"Model checksum mismatch: {path}")
        files[role] = path
    return files


def command(
    task,
    files,
    route,
    out,
    episodes,
    backend="cudagraph",
    video=False,
    render_backend=None,
):
    cmd = [
        sys.executable,
        "-m",
        "scripts.evaluate_rain",
        "--uact-checkpoint",
        str(files["uact"]),
        "--flow-checkpoint",
        str(files["flow"]),
        "--route",
        str(route),
        "--parallel-uact-streams",
        "--seed-start",
        str(task["seed_start"]),
        "--episodes",
        str(episodes),
        "--env-id",
        task["env_id"],
        "--control-mode",
        task["control_mode"],
        "--max-episode-steps",
        str(task["max_episode_steps"]),
        "--aggregation-k",
        "0.01",
        "--cache-first-query-self-attention",
        "--posthoc-diagnostic",
        "--device",
        "cuda",
        "--runtime-precision",
        "fp32",
        "--runtime-backend",
        backend,
        "--runtime-warmup-calls",
        "30",
        "--out",
        str(out),
        "--render-backend",
        render_backend or "gpu",
    ]
    if video:
        cmd.append("--capture-episode-media")
    return cmd


def main():
    tasks = load_tasks()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", *tasks], default="pickcube")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "rain")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--verify-only", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--backend", choices=["cudagraph", "eager"], default="cudagraph"
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--render-backend",
        default="gpu",
        help="ManiSkill renderer, e.g. gpu or pci:0000:00:00.0",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    selected = tasks if args.task == "all" else {args.task: tasks[args.task]}
    checked = {name: verify(task) for name, task in selected.items()}
    if args.verify_only:
        print(f"All four model files verified for {len(selected)} task(s).")
        return
    output = args.output.resolve()
    if not args.dry_run and output.exists():
        parser.error(f"Choose a new output directory: {output}")
    summary = {}
    for name, task in selected.items():
        files = checked[name]
        route_path = output / (name + ".route.json")
        out = output / name
        cmd = command(
            task,
            files,
            route_path,
            out,
            args.episodes,
            args.backend,
            args.video,
            args.render_backend,
        )
        print(shlex.join(cmd), flush=True)
        if args.dry_run:
            continue
        route = json.loads(files["route"].read_text(encoding="utf-8"))
        route["router_checkpoint"]["path"] = str(files["heads"])
        route["flow_checkpoint"]["path"] = str(files["flow"])
        output.mkdir(parents=True, exist_ok=True)
        route_path.write_text(json.dumps(route, indent=2) + "\n", encoding="utf-8")
        subprocess.run(cmd, cwd=ROOT, check=True)
        results = json.loads((out / "results.json").read_text(encoding="utf-8"))
        summary[name] = {
            "successes": results["success_once_count"],
            "episodes": args.episodes,
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    if not args.dry_run:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
