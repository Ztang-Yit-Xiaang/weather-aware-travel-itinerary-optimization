"""Exact historical inputs shared by verifier tests; not current browser proof."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath

import pytest

FIXTURES = Path(__file__).with_name("fixtures") / "recorded_runtime"


def extract_recorded_runtime(source: Path, destination: Path) -> Path:
    index = json.loads((source / "index.json").read_text(encoding="utf-8"))
    assert index["schema_version"] == "recorded-runtime-fixtures-v1"
    names = set()
    for row in index["files"]:
        name = PurePosixPath(row["path"])
        assert not name.is_absolute() and ".." not in name.parts and ":" not in str(name)
        assert "\\" not in str(name) and name.parts[:2] in {("results", "e2e"), ("results", "repairs")}
        assert str(name) not in names
        names.add(str(name))
        digest = row["sha256"]
        assert len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)
        assert row["blob"] == "blobs/" + digest + ".gz"
        packed = (source / row["blob"]).read_bytes()
        assert len(packed) == row["gzip_size_bytes"]
        assert hashlib.sha256(packed).hexdigest() == row["gzip_sha256"]
        data = gzip.decompress(packed)
        assert len(data) == row["size_bytes"] and hashlib.sha256(data).hexdigest() == digest
        target = destination / str(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    assert names
    return destination


@pytest.fixture(scope="session")
def recorded_runtime_inputs(tmp_path_factory):
    return extract_recorded_runtime(FIXTURES, tmp_path_factory.mktemp("recorded-runtime-inputs"))
