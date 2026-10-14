"""Download and verify the public RAIN pretrained models."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run import digest, load_tasks


def install(archive, manifest, selected):
    expected = {
        name: task["hashes"][role]
        for task in manifest.values()
        for role, name in task["files"].items()
    }
    wanted = {name for task in selected.values() for name in task["files"].values()}
    with tempfile.TemporaryDirectory(prefix=".rain-models-", dir=ROOT) as directory:
        stage = Path(directory)
        seen = set()
        with tarfile.open(archive, "r:gz") as tar:
            for entry in tar:
                if (
                    not entry.isfile()
                    or entry.name not in expected
                    or entry.name in seen
                ):
                    raise ValueError(f"Unexpected archive member: {entry.name}")
                seen.add(entry.name)
                target = (stage / entry.name).resolve()
                if not target.is_relative_to(stage):
                    raise ValueError("Unsafe model path")
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(entry) as source, target.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
                if digest(target) != expected[entry.name]:
                    raise ValueError(f"Model checksum mismatch: {entry.name}")
        if seen != set(expected):
            raise ValueError("Release archive is incomplete")
        for name in wanted:
            target = ROOT / name
            if target.exists() and digest(target) != expected[name]:
                raise FileExistsError(
                    f"Refusing to overwrite a different model: {target}"
                )
        for name in wanted:
            target = ROOT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.replace(stage / name, target)
    print(f"Installed and verified {len(wanted)} files.")


def main():
    tasks = load_tasks()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", *tasks], default="all")
    parser.add_argument(
        "--archive", type=Path, help="Use an already downloaded release archive"
    )
    args = parser.parse_args()
    release = json.loads((ROOT / "manifests/release.json").read_text())
    selected = tasks if args.task == "all" else {args.task: tasks[args.task]}
    with tempfile.TemporaryDirectory(prefix="rain-download-") as temp:
        archive = args.archive
        if archive is None:
            archive = Path(temp) / release["asset"]
            url = (
                f"https://github.com/{release['repository']}/releases/download/"
                f"{quote(release['tag'], safe='')}/{quote(release['asset'], safe='')}"
            )
            print(f"Downloading {release['asset']}...", flush=True)
            request = Request(url, headers={"User-Agent": "RAIN-model-downloader"})
            with urlopen(request, timeout=120) as source, archive.open("wb") as dest:
                shutil.copyfileobj(source, dest)
        if archive.stat().st_size != release["bytes"] or digest(archive) != release["sha256"]:
            raise ValueError("Release archive checksum mismatch")
        install(archive, tasks, selected)


if __name__ == "__main__":
    main()
