"""Verify and install the ManiSkill source snapshot used by the RAIN release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run import digest


def extract_source(archive: Path, destination: Path) -> int:
    count = 0
    seen = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or "\\" in member.name:
                raise ValueError(f"Unsafe source path: {member.name}")
            target = destination.joinpath(*name.parts).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Source path leaves staging directory: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or name in seen:
                raise ValueError(f"Unexpected source member: {member.name}")
            seen.add(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as input_file, target.open("wb") as output:
                shutil.copyfileobj(input_file, output)
            count += 1
    for required in ("setup.py", "LICENSE", "LICENSE-3RD-PARTY", "mani_skill/__init__.py"):
        if not (destination / required).is_file():
            raise ValueError(f"Incomplete simulator source: missing {required}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded source archive")
    parser.add_argument("--verify-only", action="store_true", help="Verify archive and source layout without installing")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "manifests/simulator.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="rain-simulator-") as temporary:
        stage = Path(temporary)
        archive = args.archive
        if archive is None:
            archive = stage / manifest["asset"]
            url = (
                f"https://github.com/{manifest['repository']}/releases/download/"
                f"{quote(manifest['tag'], safe='')}/{quote(manifest['asset'], safe='')}"
            )
            print(f"Downloading {manifest['asset']}...", flush=True)
            request = Request(url, headers={"User-Agent": "RAIN-simulator-installer"})
            with urlopen(request, timeout=120) as input_file, archive.open("wb") as output:
                shutil.copyfileobj(input_file, output)
        if archive.stat().st_size != manifest["bytes"] or digest(archive) != manifest["sha256"]:
            raise ValueError("Simulator archive checksum or size mismatch")
        source = stage / "source"
        source.mkdir()
        files = extract_source(archive, source)
        if files != manifest["files"]:
            raise ValueError("Simulator archive file count mismatch")
        print(f"Verified ManiSkill {manifest['version']} source: {files} files.", flush=True)
        if args.verify_only:
            return
        command = [sys.executable, "-m", "pip", "install", str(source)]
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
