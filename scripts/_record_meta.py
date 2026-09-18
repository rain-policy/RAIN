from __future__ import annotations
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMODULES = []


def _git(path: Path, *args: str, timeout_s: float = 30.0) -> str:
    """在 path 跑 git 命令，返回 strip 后的 stdout；失败返回 <error>。"""
    try:
        out = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
        if out.returncode != 0:
            return f"<git error: {out.stderr.strip() or 'unknown'}>"
        return out.stdout.strip()
    except FileNotFoundError:
        return "<git not found>"
    except subprocess.TimeoutExpired:
        return f"<git timeout after {timeout_s:g}s>"


def _is_dirty(path: Path, *, main_repository: bool) -> bool:
    """检查主仓库全部改动，或子模块中受版本控制文件的改动。"""
    args = ["status", "--short"]
    if main_repository:
        args.append("--ignore-submodules=dirty")
    else:
        args.append("--untracked-files=no")
    return bool(_git(path, *args))


def collect_git_info() -> dict:
    info = {
        "repo": {
            "head": _git(REPO_ROOT, "rev-parse", "HEAD"),
            "dirty": _is_dirty(REPO_ROOT, main_repository=True),
            "dirty_scope": "main_worktree_including_untracked_excluding_submodule_worktrees",
        }
    }
    info["submodules"] = {}
    for sm in SUBMODULES:
        sm_path = REPO_ROOT / sm
        if sm_path.exists():
            info["submodules"][sm] = {
                "head": _git(sm_path, "rev-parse", "HEAD"),
                "dirty": _is_dirty(sm_path, main_repository=False),
                "dirty_scope": "tracked_files_only",
            }
        else:
            info["submodules"][sm] = {"head": "<missing>", "dirty": False}
    return info


def collect_env_info() -> dict:
    env = {"python": platform.python_version(), "executable": sys.executable}
    for pkg in ("torch", "diffusers", "numpy", "gymnasium"):
        try:
            mod = __import__(pkg)
            env[pkg] = getattr(mod, "__version__", "<unknown>")
        except Exception:
            env[pkg] = "<not installed>"
    try:
        import torch

        env["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["torch_cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        env["torch_cuda_available"] = False
    return env


def _dump_config(config: dict, path_no_suffix: Path) -> str:
    """config 落盘：优先 yaml，无 pyyaml 则 json。返回实际写出的文件名。"""
    try:
        import yaml

        with open(path_no_suffix.with_suffix(".yaml"), "w") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        return path_no_suffix.with_suffix(".yaml").name
    except ImportError:
        with open(path_no_suffix.with_suffix(".json"), "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return path_no_suffix.with_suffix(".json").name


def _next_invocation_id(history_dir: Path) -> int:
    ids = []
    for path in history_dir.glob("invocation_*.json"):
        suffix = path.stem.removeprefix("invocation_")
        if suffix.isdigit():
            ids.append(int(suffix))
    return max(ids, default=0) + 1


def _archive_legacy_meta(run_dir: Path, history_dir: Path) -> None:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return
    existing = json.loads(meta_path.read_text(encoding="utf-8"))
    existing_id = existing.get("invocation_id")
    if existing_id is not None:
        snapshot = history_dir / f"invocation_{int(existing_id):06d}.json"
        if not snapshot.exists():
            raise RuntimeError(
                "current metadata references a missing invocation snapshot"
            )
        return
    config_name = existing.get("config_file")
    config_path = run_dir / str(config_name)
    if not config_name or not config_path.is_file():
        raise RuntimeError(
            "legacy metadata config is missing; refusing to overwrite provenance"
        )
    invocation_id = _next_invocation_id(history_dir)
    prefix = f"invocation_{invocation_id:06d}"
    archived_config = history_dir / f"{prefix}_config{config_path.suffix}"
    archived_config.write_bytes(config_path.read_bytes())
    archived_meta = {
        **existing,
        "invocation_id": invocation_id,
        "config_file": archived_config.name,
        "archived_from_legacy_top_level": True,
    }
    (history_dir / f"{prefix}.json").write_text(
        json.dumps(archived_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def record_meta(run_dir: str | Path, config: dict) -> Path:
    """写入当前 metadata，并为每次调用保留不可变 invocation 快照。"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    history_dir = run_dir / "metadata_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    _archive_legacy_meta(run_dir, history_dir)
    invocation_id = _next_invocation_id(history_dir)
    prefix = f"invocation_{invocation_id:06d}"
    base_meta = {
        "git": collect_git_info(),
        "env": collect_env_info(),
        "invocation_id": invocation_id,
    }
    history_config = _dump_config(config, history_dir / f"{prefix}_config")
    history_meta = {**base_meta, "config_file": history_config}
    history_path = history_dir / f"{prefix}.json"
    history_path.write_text(
        json.dumps(history_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    meta = {
        **base_meta,
        "config_file": _dump_config(config, run_dir / "config"),
        "history_record": str(history_path.relative_to(run_dir)),
    }
    meta_path = run_dir / "meta.json"
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[record_meta] wrote {meta_path}")
    print(
        f"[record_meta] repo@{meta['git']['repo']['head'][:8]} dirty={meta['git']['repo']['dirty']}"
    )
    return meta_path
