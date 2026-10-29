// Historical unit-test inputs only. Never use these to certify a fresh browser run.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";

const directory = new URL("./fixtures/recorded_runtime/", import.meta.url);
const index = JSON.parse(readFileSync(new URL("index.json", directory), "utf8"));
assert.equal(index.schema_version, "recorded-runtime-fixtures-v1");
const files = new Map(index.files.map((row) => [row.path, row]));
assert.equal(files.size, index.files.length, "fixture names must be unique");
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");

export function recordedJson(path) {
  const row = files.get(path);
  assert.ok(row, `missing recorded fixture: ${path}`);
  assert.match(row.sha256, /^[a-f0-9]{64}$/);
  assert.equal(row.blob, `blobs/${row.sha256}.gz`);
  const packed = readFileSync(new URL(row.blob, directory));
  assert.equal(packed.length, row.gzip_size_bytes);
  assert.equal(hash(packed), row.gzip_sha256);
  const bytes = gunzipSync(packed);
  assert.equal(bytes.length, row.size_bytes);
  assert.equal(hash(bytes), row.sha256);
  return JSON.parse(bytes.toString("utf8"));
}
