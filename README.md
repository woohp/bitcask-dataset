# bitcask-dataset

Basically a disk-backed `dict[bytes, bytes]` for datasets. Data is append-only,
split into shards, and mmap'd when opened. Reads don't copy.

This is mostly for training data where serialization is somebody else's problem.

## Install

```bash
uv add bitcask-dataset
```

## Example

```python
from bitcask_dataset import BitcaskDataset

ds = BitcaskDataset("my-dataset")  # creates the directory if needed
ds.extend([
    (b"image/1", image_bytes),
    (b"image/2", more_image_bytes),
])

value = ds[b"image/1"]  # read-only memoryview
value = bytes(value)     # copy if you want bytes

# insertion order is available too
value = ds.at(0)
key = ds.key_at(0)

len(ds)
b"image/1" in ds
```

Keys and values can be anything with the buffer protocol.

## API

```python
BitcaskDataset(path)
ds.extend(entries, max_shard_size=1 << 30)
ds[key]
ds.at(index)
ds.key_at(index)
len(ds)
key in ds
```

A missing or empty path is initialized. An existing non-dataset directory is an
error.

`extend` takes an iterable of `(key, value)` pairs. Duplicate keys are errors. It
only creates new shards; it never adds data to an old one. Records aren't split, so
a record over `max_shard_size` gets a shard to itself.

`ds[key]`, `at`, and `key_at` return read-only `memoryview`s. `at` and `key_at` also
support negative indexes.

## Format

```text
my-dataset/
    index.bin
    shards/
        000000.bin
        000001.bin
```

The index is little-endian `uint32` pairs:

```text
key_size, value_size, key_size, value_size, ...
```

Shards are just `key || value || key || value ...`. No headers or padding. NumPy
does a cumsum at startup and works out the shard/offset for each record.

Keys are copied into a Python dict at startup. Values stay in the mmap. Each key and
value is limited to `2**32 - 1` bytes.

## Missing stuff

No delete, overwrite, compaction, compression, or concurrent writers. Also not
actually an implementation of the Bitcask paper.

## Dev

```bash
uv sync
make format
make lint
make test
```
