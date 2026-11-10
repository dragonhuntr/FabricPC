# Golden fingerprints

`fuzzer_fingerprints.json` pins the output of the upstream-sync suite:

```bash
pytest tests/test_fuzzer_upstream_sync.py -q
pytest tests/test_fuzzer_upstream_sync.py -q --update-golden   # accept a change
```

The keys are `<config>:seed=<n>` and the values are the SHA-256 structural
fingerprints of the generated batch, in order. A mismatch means generation
output changed. That is not automatically a bug: a new FabricPC node type, a
registry change or an intentional topology change will move the fingerprints.
Review the diff, then update the golden file in the same commit. Never update it
just to make CI green.

The file is version-sensitive to the NumPy `Generator` implementation. If a
dependency bump changes it, update once and note the version in the commit
message.
