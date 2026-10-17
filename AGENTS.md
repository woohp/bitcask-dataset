# bitcask-dataset agent guide

An append-only, mmap-backed store of byte blobs for ML training data: one writer
process, many reader processes (e.g. DataLoader workers). Keep it small. Do not add
locking, concurrent writers, deletes, or compaction unless asked.

- Reads return read-only `memoryview`s in every mode, so values pass zero-copy to
  `np.frombuffer`/`torch.frombuffer` and have one exact type, even when compression
  is chosen at runtime. Do not return `bytes` or make the class generic over the
  value type.
- The on-disk format in README's "Format" section is the compatibility contract;
  change both together.

Run `make format`, `make lint` (ruff + strict mypy over src and tests), and
`make test` before finishing.
