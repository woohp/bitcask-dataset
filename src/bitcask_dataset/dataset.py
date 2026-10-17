import json
import mmap
import os
import struct
from collections.abc import Buffer, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import BinaryIO, Literal, SupportsIndex

import numpy as np
from lz4 import block

DEFAULT_MAX_SHARD_SIZE = 1 << 30
MAX_FIELD_SIZE = np.iinfo(np.uint32).max
INDEX_DTYPE = np.dtype("<u4")
MANIFEST_NAME = "manifest.json"

type Entry = tuple[Buffer, Buffer]
type Record = tuple[bytes, bytes | memoryview]
type Compression = Literal["lz4"] | None


class BitcaskDataset:
    """An append-oriented, mmap-backed collection of byte blobs."""

    @classmethod
    def create(
        cls, path: str | Path, *, keyed: bool = False, compression: Compression = "lz4"
    ) -> BitcaskDataset:
        """Create an empty dataset in a new or empty directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if any(path.iterdir()):
            raise FileExistsError(f"directory is not empty: {path}")
        _write_manifest(
            path,
            keyed=keyed,
            compression=_validate_compression(compression),
            shard_count=0,
        )
        return cls(path)

    def __init__(self, path: str | Path) -> None:
        """Open an existing dataset."""
        self.path = Path(path)
        self._load()

    def extend(
        self,
        entries: Iterable[Buffer | Entry],
        *,
        max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
    ) -> None:
        """Add values, or unique key/value pairs for a keyed dataset, as new shards.

        Keys and values must be C-contiguous buffers. Values are compressed in
        parallel, so their buffers must not be modified after they are yielded.
        """
        if max_shard_size <= 0:
            raise ValueError("max_shard_size must be positive")

        # Pick up shards committed through another handle, so they aren't overwritten.
        self._add_shards(_read_manifest(self.path)[2])
        records = _records(entries, keyed=self.keyed, existing_keys=self._keys)
        workers = os.process_cpu_count() or 1
        with ThreadPoolExecutor(workers) as pool:
            if self.compression == "lz4":
                # lz4 releases the GIL; buffersize bounds how many records
                # are held in memory.
                records = pool.map(_compress, records, buffersize=2 * workers)
            shard_count = _write_shards(
                self.path,
                records,
                keyed=self.keyed,
                first_shard_id=len(self._shards),
                max_shard_size=max_shard_size,
            )
        _write_manifest(
            self.path,
            keyed=self.keyed,
            compression=self.compression,
            shard_count=shard_count,
        )
        self._add_shards(shard_count)

    def __len__(self) -> int:
        return self._sizes.shape[0]

    def __contains__(self, key: object) -> bool:
        return self.keyed and isinstance(key, Buffer) and _as_key(key) in self._keys

    def __getitem__(self, key: Buffer) -> memoryview:
        if not self.keyed:
            raise TypeError("key lookup is disabled for this dataset")
        return self.at(self._keys[_as_key(key)])

    def at(self, index: SupportsIndex) -> memoryview:
        """Return a value by insertion position."""
        key_size = int(self._sizes[index, 0]) if self.keyed else 0
        value_size = int(self._sizes[index, -1])
        offset = int(self._record_offsets[index]) + key_size
        value = self._shards[int(self._shard_ids[index])][offset : offset + value_size]
        if self.compression == "lz4" and value_size:
            return memoryview(block.decompress(value))
        return value

    def key_at(self, index: SupportsIndex) -> memoryview:
        """Return a key by insertion position."""
        if not self.keyed:
            raise TypeError("keys are disabled for this dataset")
        offset = int(self._record_offsets[index])
        return self._shards[int(self._shard_ids[index])][
            offset : offset + int(self._sizes[index, 0])
        ]

    def __getstate__(self) -> Path:
        return self.path

    def __setstate__(self, state: Path) -> None:
        self.path = state
        self._load()

    def _load(self) -> None:
        self.keyed, self.compression, shard_count = _read_manifest(self.path)
        self._shards: list[memoryview] = []
        self._sizes = np.empty((0, 2 if self.keyed else 1), dtype=INDEX_DTYPE)
        self._shard_ids = np.empty(0, dtype=np.uint32)
        self._record_offsets = np.empty(0, dtype=np.uint64)
        self._keys: dict[bytes, int] = {}
        self._add_shards(shard_count)

    def _add_shards(self, shard_count: int) -> None:
        if shard_count == len(self._shards):
            return
        shards: list[memoryview] = []
        sizes = [self._sizes]
        shard_ids = [self._shard_ids]
        record_offsets = [self._record_offsets]
        new_keys: list[bytes] = []
        for shard_id in range(len(self._shards), shard_count):
            data_path, index_path = _shard_paths(self.path, shard_id)
            shard_sizes = np.fromfile(index_path, dtype=INDEX_DTYPE).reshape(
                -1, self._sizes.shape[1]
            )
            record_sizes = shard_sizes.sum(axis=1, dtype=np.uint64)
            shard = _mmap_readonly(data_path)
            if int(record_sizes.sum()) != len(shard):
                raise ValueError(f"shard size does not match its index: {data_path}")

            offsets = np.cumsum(record_sizes, dtype=np.uint64) - record_sizes
            if self.keyed:
                key_sizes = shard_sizes[:, 0].tolist()
                new_keys += [
                    bytes(shard[offset : offset + key_size])
                    for offset, key_size in zip(offsets.tolist(), key_sizes)
                ]
            shards.append(shard)
            sizes.append(shard_sizes)
            shard_ids.append(np.full(len(shard_sizes), shard_id, dtype=np.uint32))
            record_offsets.append(offsets)

        # Validate everything before changing any state, so a failure leaves the
        # handle as it was.
        keys = {key: index for index, key in enumerate(new_keys, len(self))}
        if len(keys) != len(new_keys) or not self._keys.keys().isdisjoint(keys):
            raise ValueError("duplicate keys in dataset")
        self._shards += shards
        self._sizes = np.concatenate(sizes)
        self._shard_ids = np.concatenate(shard_ids)
        self._record_offsets = np.concatenate(record_offsets)
        self._keys.update(keys)


def _validate_compression(compression: object) -> Compression:
    if compression == "lz4":
        return "lz4"
    if compression is None:
        return None
    raise ValueError(f"unsupported compression: {compression!r}")


def _write_manifest(
    path: Path, *, keyed: bool, compression: Compression, shard_count: int
) -> None:
    manifest = {"keyed": keyed, "compression": compression, "shards": shard_count}
    temporary = path / f".{MANIFEST_NAME}.tmp"
    with temporary.open("wb") as file:
        file.write(json.dumps(manifest).encode())
        _finish_files((file,))
    # New shard files must be durable before the manifest rename, which commits them.
    _fsync_directory(path)
    os.replace(temporary, path / MANIFEST_NAME)
    _fsync_directory(path)


def _read_manifest(path: Path) -> tuple[bool, Compression, int]:
    manifest_path = path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    keyed, compression, shard_count = (
        manifest["keyed"],
        manifest["compression"],
        manifest["shards"],
    )
    if (
        not isinstance(keyed, bool)
        or not isinstance(shard_count, int)
        or shard_count < 0
    ):
        raise ValueError(f"invalid manifest: {manifest_path}")
    return keyed, _validate_compression(compression), shard_count


def _records(
    entries: Iterable[Buffer | Entry],
    *,
    keyed: bool,
    existing_keys: dict[bytes, int],
) -> Iterator[Record]:
    new_keys: set[bytes] = set()
    for entry in entries:
        if not keyed:
            if not isinstance(entry, Buffer):
                raise TypeError("keyless datasets require an iterable of buffers")
            yield b"", _as_bytes_view(entry)
            continue

        if isinstance(entry, Buffer):
            raise TypeError("keyed datasets require an iterable of (key, value) pairs")
        raw_key, value = entry
        key = _as_key(raw_key)
        if key in existing_keys or key in new_keys:
            raise ValueError(f"duplicate key: {key!r}")
        new_keys.add(key)
        yield key, _as_bytes_view(value)


def _compress(record: Record) -> Record:
    key, value = record
    if not len(value):
        return record
    return key, block.compress(value, mode="high_compression", store_size=True)


def _write_shards(
    path: Path,
    records: Iterable[Record],
    *,
    keyed: bool,
    first_shard_id: int,
    max_shard_size: int,
) -> int:
    """Write records to new, uncommitted shard files and return the new shard count."""
    shard_count = first_shard_id
    written: list[Path] = []
    files: tuple[BinaryIO, BinaryIO] | None = None
    shard_size = 0
    try:
        for key, value in records:
            if len(key) > MAX_FIELD_SIZE or len(value) > MAX_FIELD_SIZE:
                raise ValueError("keys and encoded values must fit in uint32")

            record_size = len(key) + len(value)
            if files is None or (
                shard_size and shard_size + record_size > max_shard_size
            ):
                if files is not None:
                    _finish_files(files)
                data_path, index_path = _shard_paths(path, shard_count)
                written += [data_path, index_path]
                files = data_path.open("wb"), index_path.open("wb")
                shard_count += 1
                shard_size = 0

            data_file, index_file = files
            data_file.write(key)
            data_file.write(value)
            shard_size += record_size
            if keyed:
                index_file.write(struct.pack("<II", len(key), len(value)))
            else:
                index_file.write(struct.pack("<I", len(value)))

        if files is not None:
            _finish_files(files)
    except BaseException:
        for file in files or ():
            file.close()
        for written_path in written:
            written_path.unlink(missing_ok=True)
        raise
    return shard_count


def _shard_paths(path: Path, shard_id: int) -> tuple[Path, Path]:
    return path / f"{shard_id:06d}.bin", path / f"{shard_id:06d}.idx"


def _mmap_readonly(path: Path) -> memoryview:
    if path.stat().st_size == 0:
        return memoryview(b"")
    with path.open("rb") as file:
        return memoryview(mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ))


def _as_key(value: Buffer) -> bytes:
    return bytes(_as_bytes_view(value))


def _as_bytes_view(value: Buffer) -> memoryview:
    try:
        view = memoryview(value)
    except TypeError:
        raise TypeError("keys and values must support the buffer protocol") from None
    return view.cast("B")


def _finish_files(files: Iterable[BinaryIO]) -> None:
    for file in files:
        file.flush()
        os.fsync(file.fileno())
        file.close()


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
