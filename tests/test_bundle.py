import json
from pathlib import Path
import tarfile

import pytest

from kev_service.catalog import MODELS, UPSTREAM_COMMIT
from kev_service.download import pack, sha256, verify


def bundle(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"trusted weights")
    (source / "metadata.lock").write_bytes(b"")
    (source / "bundle.json").write_text(json.dumps({
        "model_id": "kev-0.8b", **MODELS["kev-0.8b"], "upstream_commit": UPSTREAM_COMMIT,
        "files": {name: sha256(source / name) for name in ("weights.bin", "metadata.lock")}}))
    return source


def test_portable_bundle_and_checksum(tmp_path):
    source = bundle(tmp_path)
    archive = tmp_path / "model.tar"
    pack(source, archive)
    destination = tmp_path / "unpacked"
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter="data")
    assert verify(destination)["model_id"] == "kev-0.8b"
    (destination / "weights.bin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        verify(destination)


def test_manifest_cannot_escape_bundle(tmp_path):
    source = bundle(tmp_path)
    manifest = json.loads((source / "bundle.json").read_text())
    outside = tmp_path / "outside"
    outside.write_bytes(b"other")
    manifest["files"] = {"../outside": sha256(outside)}
    (source / "bundle.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify(source)


def test_archive_extraction_blocks_traversal(tmp_path):
    archive = tmp_path / "evil.tar"
    import io
    with tarfile.open(archive, "w") as tar:
        member = tarfile.TarInfo("../escape")
        member.size = 4
        tar.addfile(member, io.BytesIO(b"evil"))
    with tarfile.open(archive) as tar, pytest.raises(tarfile.FilterError):
        tar.extractall(tmp_path / "unpacked", filter="data")
    assert not Path(tmp_path / "escape").exists()
