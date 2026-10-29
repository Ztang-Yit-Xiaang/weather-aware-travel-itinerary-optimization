# Recorded runtime test inputs

These compressed files are exact copies of historical synthetic-demo receipts
used by Python verifier and Node presentation tests. The index records each
original path, byte count, SHA-256, and compressed-file identity. Readers check
both compressed and decompressed bytes. Gzip blobs are deduplicated by content.

They are unit-test fixtures, not proof that the current website works. Historical
schema, capability, and source mismatches are intentionally retained as negative
examples. The selection-generation fixture exercises that specific source guard;
it does not certify the historical fourteen-case browser bundle.

Fresh acceptance still requires the real-server browser suite and its independent
verifier. Original result directories are not modified or required by these tests.
