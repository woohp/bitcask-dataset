# bitcask-dataset

A disk-backed blob store that's a good fit if:

- you mostly do random-access reads, e.g. ML training data,
- values are biggish blobs (KiB to MiB) and keys are small,
- you add data in batches and never update or delete.

Each `extend` writes new shard files. Opening a dataset mmaps the shards as one logical
file, so lookups are fast and efficient.
Values are optionally LZ4-compressed (on by default).

Because old shards never change, keeping a copy elsewhere (say, S3) after an `extend`
means uploading only the new shards and the manifest, and never the whole dataset.

Not a good fit if you need deletes, overwrites, concurrent writers, or more keys
than what your RAM can fit.

## Install

```bash
pip install bitcask-dataset
```

## Example

```python
from bitcask_dataset import BitcaskDataset

ds = BitcaskDataset.create("my-dataset", keyed=True)
ds.extend([
    (b"image/1", image_bytes),
    (b"image/2", more_image_bytes),
])

ds = BitcaskDataset("my-dataset")  # settings come from the manifest
value = ds[b"image/1"]  # read-only memoryview of the decompressed bytes

# insertion order is available too
value = ds.at(0)
key = ds.key_at(0)

len(ds)
b"image/1" in ds
```

Keys are optional:

```python
ds = BitcaskDataset.create("my-keyless-dataset")
ds.extend([image_bytes, more_image_bytes])
value = ds.at(0)
```

Keys and values can be any C-contiguous buffer (`bytes`, `bytearray`, a contiguous NumPy
array, ...).

## API

```python
BitcaskDataset.create(path, keyed=False, compression="lz4", max_shard_size=1 << 30)
BitcaskDataset(path)
ds.extend(entries)
ds[key]
ds.at(index)
ds.key_at(index)
len(ds)
key in ds
```

`create` needs a missing or empty directory. `BitcaskDataset(path)` opens an existing
dataset; `keyed`, `compression`, and `max_shard_size` are fixed at creation and read
from the manifest.

In keyed mode, `extend` takes `(key, value)` pairs and duplicate keys are errors. In
keyless mode it takes values directly. Each `extend` is all-or-nothing and only
creates new shards. Records aren't split, so a record over `max_shard_size` gets a
shard to itself. Values are compressed on a thread pool, so don't modify a value's
buffer after yielding it to `extend`.

Keys and values come back as read-only `memoryview`s, which `np.frombuffer` and
`torch.frombuffer` can use without copying. Non-empty compressed values are views of
freshly decompressed `bytes` (`value.obj`). With `compression=None`, values are read
straight from the mmap with zero copies. Keys are always uncompressed. `at` and `key_at`
support negative indexes. Key lookup and `key_at` raise `TypeError` in keyless mode.

## Format

```text
my-dataset/
    manifest.json
    000000.bin
    000000.idx
    000001.bin
    000001.idx
```

`manifest.json` holds the settings and how many shards are committed:

```json
{"keyed": true, "compression": "lz4", "max_shard_size": 1073741824, "shards": 2}
```

Each `.idx` lists its shard's records as little-endian `uint32` sizes: `key_size,
value_size` pairs when keyed, one stored-value size per record when keyless. Keyed
shards are `key || value || key || value ...`; keyless shards only contain values.
There are no record headers or padding. Values are independent LZ4 blocks using
high-compression mode, with the original size in each block. Empty values stay empty.

Rewriting the manifest commits an `extend`, so shard files it doesn't list are
leftovers from an interrupted `extend`; they're ignored and overwritten later. When
copying a dataset, copy the manifest last.

Keyed datasets copy keys into a Python dict in each process that opens them (e.g. every
DataLoader worker). Each key and stored value is limited to `2**32 - 1` bytes; with LZ4,
uncompressed values are also limited to about 2 GiB, the LZ4 block limit.

## Missing stuff

No delete, overwrite, compaction, or concurrent writers. Also not actually an
implementation of the Bitcask paper, though per-shard indexes are close to its hint
files.

## Dev

```bash
uv sync
make format
make lint
make test
```
