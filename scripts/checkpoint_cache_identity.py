from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any, Mapping


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_cache_identity(
    state: Mapping[str, Any], cache_state: Mapping[str, Any], cache_path: Path
) -> None:
    """Validate either a frozen training cache or a post-joint U-ACT cache.

    Ordinary frozen-refiner checkpoints pin the cache file directly. Joint
    deployment checkpoints necessarily precede the post-joint causal cache;
    those pin the verified deployment U-ACT that the new cache must reference.
    """
    direct = state.get("cache", {})
    if isinstance(direct, Mapping) and direct.get("sha256"):
        if str(direct["sha256"]) != sha256(cache_path):
            raise RuntimeError("refiner and causal cache hashes disagree")
        return
    joint_source = state.get("joint_source", {})
    uact_source = state.get("uact_source", {})
    cache_uact_source = cache_state.get("uact_source", {})
    if not all(
        (
            isinstance(item, Mapping)
            for item in (joint_source, uact_source, cache_uact_source)
        )
    ):
        raise RuntimeError("joint deployment cache provenance is malformed")
    if not joint_source.get("path") or not joint_source.get("sha256"):
        raise RuntimeError("joint deployment has no verifiable joint source")
    joint_path = Path(str(joint_source["path"])).expanduser().resolve()
    if not joint_path.is_file() or sha256(joint_path) != str(joint_source["sha256"]):
        raise RuntimeError(
            "joint source checkpoint hash disagrees with deployment provenance"
        )
    if not uact_source.get("path") or not uact_source.get("sha256"):
        raise RuntimeError("joint deployment has no verifiable U-ACT source")
    uact_path = Path(str(uact_source["path"])).expanduser().resolve()
    if not uact_path.is_file() or sha256(uact_path) != str(uact_source["sha256"]):
        raise RuntimeError("deployment U-ACT hash disagrees with refiner provenance")
    if str(cache_uact_source.get("sha256", "")) != str(uact_source["sha256"]):
        raise RuntimeError("post-joint cache and deployed U-ACT hashes disagree")
