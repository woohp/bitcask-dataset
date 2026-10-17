# bitcask-dataset

Basically a disk-backed collection of byte blobs, optionally keyed by bytes. Data is
append-only, split into shards, LZ4-compressed, and mmap'd when opened.

This is mostly for training data where serialization is somebody else's problem.

## Install

```bash
pip install bitcask-dataset
```

## Example

```python
from bitcask_dataset import BitcaskDataset

ds = BitcaskDataset("my-dataset", keyed=True)  # creates the directory if needed
ds.extend([
    (b"image/1", image_bytes),
    (b"image/2", more_image_bytes),
])

value = ds[b"image/1"]  # read-only memoryview of the decompressed bytes

# insertion order is available too
value = ds.at(0)
key = ds.key_at(0)

len(ds)
b"image/1" in ds
```

Keys are optional:

```python
ds = BitcaskDataset("my-keyless-dataset")
ds.extend([image_bytes, more_image_bytes])
value = ds.at(0)
```

Keys and values can be anything with the buffer protocol.

## API

```python
BitcaskDataset(path, keyed=False, compression="lz4")
ds.extend(entries, max_shard_size=1 << 30)
ds[key]
ds.at(index)
ds.key_at(index)
len(ds)
key in ds
```

A missing or empty path is initialized. An existing non-dataset directory is an
error.

In keyed mode, `extend` takes `(key, value)` pairs and duplicate keys are errors. In
keyless mode it takes values directly. It only creates new shards; it never adds data
to an old one. Records aren't split, so a record over `max_shard_size` gets a shard
to itself.

Keys and values come back as read-only `memoryview`s, which `np.frombuffer` and
`torch.frombuffer` can use without copying. Compressed values are views of freshly
decompressed `bytes` (`value.obj`). Pass `compression=None` to store values
uncompressed and read them straight from the mmap with zero copies. Keys are always
uncompressed. `at` and `key_at` support negative indexes. Key lookup and `key_at`
raise `TypeError` in keyless mode.

There is no metadata file yet, so remember to use the same `keyed` and `compression`
arguments every time you open a dataset. The defaults are `keyed=False` and
`compression="lz4"`. Opening a dataset with the wrong settings may misread it.

## Format

```text
my-dataset/
    index.bin
    shards/
        000000.bin
        000001.bin
```

The keyed index is little-endian `uint32` pairs:

```text
key_size, value_size, key_size, value_size, ...
```

The keyless index is one `uint32` stored-value size per item. Keyed shards are
`key || value || key || value ...`; keyless shards only contain values. Values are
independent LZ4 blocks using high-compression mode, with the original size in each
block. Empty values stay empty. There are no record headers or padding. NumPy does a
cumsum at startup and works out the shard/offset for each record.

Keyed datasets copy keys into a Python dict at startup. Each key and uncompressed
value is limited to `2**32 - 1` bytes.

## Missing stuff

An interrupted `extend` can leave extra shard files behind. They're ignored when
opening and replaced by the next `extend`.

No delete, overwrite, compaction, or concurrent writers. Also not
actually an implementation of the Bitcask paper.

## Dev

```bash
uv sync
make format
make lint
make test
```
