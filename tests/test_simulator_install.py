import io
from pathlib import Path
import tarfile

import pytest

from tools.install_simulator import extract_source


def write_archive(path, files):
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))


def test_source_installer_rejects_path_traversal(tmp_path):
    archive = tmp_path / "source.tar.gz"
    write_archive(archive, {"../outside.py": b"untrusted"})
    with pytest.raises(ValueError, match="Unsafe source path"):
        extract_source(archive, tmp_path / "stage")
    assert not (tmp_path / "outside.py").exists()


def test_source_installer_rejects_link_members(tmp_path):
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as source:
        link = tarfile.TarInfo("linked")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        source.addfile(link)
    with pytest.raises(ValueError, match="Unexpected source member"):
        extract_source(archive, tmp_path / "stage")


def test_source_installer_checks_required_build_files(tmp_path):
    archive = tmp_path / "source.tar.gz"
    write_archive(archive, {"./setup.py": b"# setup"})
    with pytest.raises(ValueError, match="Incomplete simulator source"):
        extract_source(archive, tmp_path / "stage")


def test_source_installer_preserves_verified_source_bytes(tmp_path):
    archive = tmp_path / "source.tar.gz"
    files = {"./setup.py": b"# setup", "./LICENSE": b"license", "./LICENSE-3RD-PARTY": b"notices", "./mani_skill/__init__.py": b"__version__ = '3.0.1'"}
    write_archive(archive, files)
    stage = tmp_path / "stage"
    assert extract_source(archive, stage) == len(files)
    for name, content in files.items():
        assert (stage / name).read_bytes() == content
