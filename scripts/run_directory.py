from __future__ import annotations
from pathlib import Path


def prepare_run_directory(run_dir: str | Path) -> Path:
    """Reserve a run directory without accepting a previous training artifact.

    ``run_direct_gpu_task.sh`` writes its launch provenance and active marker
    before it invokes the task launcher.  Only those active-run files may be
    present; every completed, failed, or trainer artifact remains fail-closed.
    """
    path = Path(run_dir)
    if not path.exists():
        path.mkdir(parents=True)
        return path
    if not path.is_dir():
        raise FileExistsError(f"run path is not a directory: {path}")
    active_direct_files = {
        "direct_launches",
        "direct.log",
        "direct.running",
        "preflight.json",
    }
    present = {item.name for item in path.iterdir()}
    unexpected = sorted(present - active_direct_files)
    if unexpected:
        raise FileExistsError(
            f"refusing to reuse run directory with training artifacts: {path} (found: {', '.join(unexpected)})"
        )
    provenance_dir = path / "direct_launches"
    if provenance_dir.exists() and (not provenance_dir.is_dir()):
        raise FileExistsError(
            f"run provenance path is not a directory: {provenance_dir}"
        )
    direct_markers = {"direct.log", "direct.running"}
    if present & direct_markers and (not direct_markers <= present):
        raise FileExistsError(
            f"incomplete direct-GPU active marker set; refusing to reuse run directory: {path}"
        )
    if "preflight.json" in present and "direct.running" not in present:
        raise FileExistsError(
            f"preflight artifact is accepted only during an active direct-GPU launch: {path}"
        )
    return path
