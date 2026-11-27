"""Integrity checks for portable historical data, not runtime acceptance."""

from __future__ import annotations

import gzip
import hashlib
import json

import pytest
from recorded_runtime_fixture import extract_recorded_runtime

pytest_plugins = ["recorded_runtime_fixture"]


@pytest.mark.parametrize("fault", [None, "path_escape", "blob_escape", "size", "hash", "gzip_hash", "duplicate"])
def test_recorded_inputs_preserve_bytes_and_reject_corruption(tmp_path, fault):
    source = tmp_path / "fixtures"
    (source / "blobs").mkdir(parents=True)
    data = b'{"historical":true}\n'
    packed = gzip.compress(data, mtime=0)
    digest = hashlib.sha256(data).hexdigest()
    row = {
        "path": "results/e2e/example/receipt.json",
        "sha256": digest,
        "size_bytes": len(data),
        "blob": "blobs/" + digest + ".gz",
        "gzip_sha256": hashlib.sha256(packed).hexdigest(),
        "gzip_size_bytes": len(packed),
    }
    (source / row["blob"]).write_bytes(packed)
    if fault == "path_escape":
        row["path"] = "results/e2e/../../../escaped.json"
    elif fault == "blob_escape":
        row["blob"] = "../../escaped.gz"
    elif fault == "size":
        row["size_bytes"] += 1
    elif fault == "hash":
        packed = gzip.compress(b'{"historical":null}\n', mtime=0)
        (source / row["blob"]).write_bytes(packed)
        row.update(gzip_sha256=hashlib.sha256(packed).hexdigest(), gzip_size_bytes=len(packed))
    elif fault == "gzip_hash":
        row["gzip_sha256"] = "0" * 64
    index = {"schema_version": "recorded-runtime-fixtures-v1", "files": [row] * (2 if fault == "duplicate" else 1)}
    (source / "index.json").write_text(json.dumps(index))
    destination = tmp_path / "extracted"
    if fault is None:
        assert extract_recorded_runtime(source, destination) == destination
        assert (destination / row["path"]).read_bytes() == data
    else:
        with pytest.raises(AssertionError):
            extract_recorded_runtime(source, destination)
    assert not (tmp_path / "escaped.json").exists()


def test_decompressed_historical_inputs_pass_evidence_credential_scan(recorded_runtime_inputs):
    from scripts.verify_e2e1_live_copilot_chat import _verify_text_security

    _verify_text_security(recorded_runtime_inputs)
