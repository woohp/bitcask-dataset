from __future__ import annotations

import mmap
import os
import struct
import tempfile
from collections.abc import Buffer, Iterable
from pathlib import Path
from typing import BinaryIO, Literal, SupportsIndex, cast, overload

import numpy as np
import numpy.typing as npt
from lz4 import block

DEFAULT_MAX_SHARD_SIZE = 1 << 30
MAX_FIELD_SIZE = np.iinfo(np.uint32).max
INDEX_DTYPE = np.dtype("<u4")
EMPTY_VIEW = memoryview(b"")

type Entry = tuple[Buffer, Buffer]
type Compression = Literal["lz4"] | None
type UIntArray = npt.NDArray[np.unsignedinteger]
type Shard = mmap.mmap | memoryview


class BitcaskDataset[ValueT: Buffer]:
    """An append-oriented, mmap-backed collection of byte blobs."""

    @overload
    def __init__(
        self: BitcaskDataset[bytes],
        path: str | Path,
        *,
        keyed: bool = False,
        compression: Literal["lz4"] = "lz4",
    ) -> None: ...

    @overload
    def __init__(
        self: BitcaskDataset[memoryview],
        path: str | Path,
        *,
        keyed: bool = False,
        compression: None,
    ) -> None: ...

    @overload
    def __init__(
        self: BitcaskDataset[bytes | memoryview],
        path: str | Path,
        *,
        keyed: bool = False,
        compression: Compression,
    ) -> None: ...

    def __init__(
        self,
        path: str | Path,
        *,
        keyed: bool = False,
        compression: Compression = "lz4",
    ) -> None:
        self.path = Path(path)
        self.keyed = keyed
        self._compression = _validate_compression(compression)
        if not self.path.exists() or not any(self.path.iterdir()):
            _initialize_empty_dataset(self.path)
        self._load()

    def extend(
        self,
        entries: Iterable[Buffer | Entry],
        *,
        max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
    ) -> None:
        """Add values, or unique key/value pairs for a keyed dataset."""
        if max_shard_size <= 0:
            raise ValueError("max_shard_size must be positive")

        _extend_dataset(
            self.path,
            entries,
            keyed=self.keyed,
            compression=self.compression,
            existing_keys=self._keys,
            first_shard_id=len(self._shards),
            max_shard_size=max_shard_size,
        )
        self._load()

    def __len__(self) -> int:
        return self._sizes.shape[0]

    def __contains__(self, key: object) -> bool:
        return self.keyed and isinstance(key, Buffer) and _as_key(key) in self._keys

    @property
    def compression(self) -> Compression:
        return self._compression

    def __getitem__(self, key: Buffer) -> ValueT:
        if not self.keyed:
            raise TypeError("key lookup is disabled for this dataset")
        normalized = _as_key(key)
        try:
            index = self._keys[normalized]
        except KeyError:
            raise KeyError(normalized) from None
        return self._value_at(index)

    def at(self, index: SupportsIndex) -> ValueT:
        """Return a value by insertion position."""
        return self._value_at(index)

    def key_at(self, index: SupportsIndex) -> memoryview:
        """Return a key by insertion position."""
        if not self.keyed:
            raise TypeError("keys are disabled for this dataset")
        return self._key_at(index)

    def __getstate__(self) -> tuple[Path, bool, Compression]:
        return self.path, self.keyed, self.compression

    def __setstate__(self, state: tuple[Path, bool, Compression]) -> None:
        self.path, self.keyed, self._compression = state
        self._load()

    def _load(self) -> None:
        index_path = self.path / "index.bin"
        columns = 2 if self.keyed else 1
        if index_path.stat().st_size:
            raw_sizes = np.memmap(index_path, dtype=INDEX_DTYPE, mode="r")
            self._sizes = raw_sizes.reshape(-1, columns)
        else:
            self._sizes = np.empty((0, columns), dtype=INDEX_DTYPE)

        shard_paths = _shard_paths(self.path / "shards")
        shard_sizes = np.asarray([path.stat().st_size for path in shard_paths], dtype=np.uint64)
        expected_size = int(self._sizes.sum(dtype=np.uint64))
        actual_size = int(shard_sizes.sum())
        if actual_size != expected_size:
            raise ValueError(f"shard data size is {actual_size}, expected {expected_size}")

        self._shards = [_mmap_readonly(path) for path in shard_paths]
        self._shard_ids, self._record_offsets = _build_locations(self._sizes, shard_sizes)
        self._keys: dict[bytes, int] = {}
        if self.keyed:
            self._keys = {bytes(self._key_at(index)): index for index in range(len(self))}
            if len(self._keys) != len(self):
                raise ValueError("duplicate keys in dataset")

    def _key_at(self, index: SupportsIndex) -> memoryview:
        key_size = int(self._sizes[index, 0])
        if key_size == 0:
            return EMPTY_VIEW

        shard_id = int(self._shard_ids[index])
        offset = int(self._record_offsets[index])
        return memoryview(self._shards[shard_id])[offset : offset + key_size]

    def _value_at(self, index: SupportsIndex) -> ValueT:
        key_size = int(self._sizes[index, 0]) if self.keyed else 0
        value_column = 1 if self.keyed else 0
        value_size = int(self._sizes[index, value_column])
        if value_size == 0:
            empty = b"" if self.compression else EMPTY_VIEW
            return cast(ValueT, empty)

        shard_id = int(self._shard_ids[index])
        offset = int(self._record_offsets[index]) + key_size
        value = memoryview(self._shards[shard_id])[offset : offset + value_size]
        if self.compression == "lz4":
            decompressed = block.decompress(value)  # type: ignore[attr-defined]
            return cast(ValueT, decompressed)
        return cast(ValueT, value)


def _validate_compression(compression: str | None) -> Compression:
    if compression == "lz4":
        return "lz4"
    if compression is None:
        return None
    raise ValueError(f"unsupported compression: {compression!r}")


def _initialize_empty_dataset(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "shards").mkdir()
    (path / "index.bin").touch()


def _extend_dataset(
    path: Path,
    entries: Iterable[Buffer | Entry],
    *,
    keyed: bool,
    compression: Compression,
    existing_keys: dict[bytes, int],
    first_shard_id: int,
    max_shard_size: int,
) -> None:
    shards_path = path / "shards"
    temporary_shards: list[tuple[Path, Path]] = []
    new_keys: set[bytes] = set()
    shard_file = None
    shard_size = 0
    next_shard_id = first_shard_id

    index_fd, index_name = tempfile.mkstemp(prefix=".index-", dir=path)
    index_temp = Path(index_name)

    try:
        with os.fdopen(index_fd, "wb") as index_file:
            with (path / "index.bin").open("rb") as old_index:
                _copy_file(old_index, index_file)

            for entry in entries:
                if keyed:
                    key, value = _entry_views(entry)
                    key_copy = bytes(key)
                    if key_copy in existing_keys or key_copy in new_keys:
                        raise ValueError(f"duplicate key: {key_copy!r}")
                    new_keys.add(key_copy)
                    if key.nbytes > MAX_FIELD_SIZE:
                        raise ValueError("key must fit in uint32")
                else:
                    if not isinstance(entry, Buffer):
                        raise TypeError("keyless datasets require an iterable of buffers")
                    key = EMPTY_VIEW
                    value = _as_bytes_view(entry)

                stored_value: bytes | memoryview
                if compression == "lz4" and value.nbytes:
                    stored_value = block.compress(  # type: ignore[attr-defined]
                        value, mode="high_compression", store_size=True
                    )
                else:
                    stored_value = value
                stored_value_size = len(stored_value)
                if stored_value_size > MAX_FIELD_SIZE:
                    raise ValueError("encoded value must fit in uint32")

                record_size = key.nbytes + stored_value_size
                if shard_file is None or (shard_size and shard_size + record_size > max_shard_size):
                    if shard_file is not None:
                        _finish_file(shard_file)
                    final_path = shards_path / f"{next_shard_id:06d}.bin"
                    fd, temp_name = tempfile.mkstemp(prefix=f".{next_shard_id:06d}-", dir=shards_path)
                    shard_file = os.fdopen(fd, "wb")
                    temporary_shards.append((Path(temp_name), final_path))
                    next_shard_id += 1
                    shard_size = 0

                shard_file.write(key)
                shard_file.write(stored_value)
                shard_size += record_size
                if keyed:
                    index_file.write(struct.pack("<II", key.nbytes, stored_value_size))
                else:
                    index_file.write(struct.pack("<I", stored_value_size))

            if shard_file is not None:
                _finish_file(shard_file)
                shard_file = None

            index_file.flush()
            os.fsync(index_file.fileno())

        for temporary, final in temporary_shards:
            os.replace(temporary, final)
        os.replace(index_temp, path / "index.bin")
    except BaseException:
        if shard_file is not None:
            shard_file.close()
        index_temp.unlink(missing_ok=True)
        for temporary, _ in temporary_shards:
            temporary.unlink(missing_ok=True)
        raise


def _build_locations(
    sizes: npt.NDArray[np.uint32],
    shard_sizes: npt.NDArray[np.uint64],
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint64]]:
    record_sizes = sizes.sum(axis=1, dtype=np.uint64)
    global_starts = np.cumsum(record_sizes) - record_sizes
    shard_ends = np.cumsum(shard_sizes)

    shard_ids = np.zeros(record_sizes.size, dtype=np.uint32)
    record_offsets = np.zeros(record_sizes.size, dtype=np.uint64)
    nonempty = record_sizes != 0
    resolved_ids = np.searchsorted(shard_ends, global_starts[nonempty], side="right")
    resolved_offsets = global_starts[nonempty] - (shard_ends - shard_sizes)[resolved_ids]
    if np.any(resolved_offsets + record_sizes[nonempty] > shard_sizes[resolved_ids]):
        raise ValueError("an indexed record crosses a shard boundary")

    shard_ids[nonempty] = resolved_ids
    record_offsets[nonempty] = resolved_offsets
    return shard_ids, record_offsets


def _shard_paths(path: Path) -> list[Path]:
    numbered: list[tuple[int, Path]] = []
    for candidate in path.iterdir():
        if candidate.name.startswith("."):
            continue
        if candidate.suffix != ".bin" or not candidate.stem.isdecimal():
            raise ValueError(f"invalid shard filename: {candidate.name}")
        numbered.append((int(candidate.stem), candidate))

    numbered.sort()
    if [number for number, _ in numbered] != list(range(len(numbered))):
        raise ValueError("shard numbers must be contiguous starting at zero")
    return [candidate for _, candidate in numbered]


def _mmap_readonly(path: Path) -> Shard:
    if path.stat().st_size == 0:
        return EMPTY_VIEW
    with path.open("rb") as file:
        return mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ)


def _entry_views(entry: Buffer | Entry) -> tuple[memoryview, memoryview]:
    if isinstance(entry, Buffer):
        raise TypeError("keyed datasets require an iterable of (key, value) pairs")
    key, value = entry
    return _as_bytes_view(key), _as_bytes_view(value)


def _as_key(value: Buffer) -> bytes:
    return bytes(_as_bytes_view(value))


def _as_bytes_view(value: Buffer) -> memoryview:
    try:
        view = memoryview(value)
    except TypeError:
        raise TypeError("keys and values must support the buffer protocol") from None
    return view.cast("B")


def _copy_file(source: BinaryIO, destination: BinaryIO) -> None:
    while chunk := source.read(1024 * 1024):
        destination.write(chunk)


def _finish_file(file: BinaryIO) -> None:
    file.flush()
    os.fsync(file.fileno())
    file.close()
