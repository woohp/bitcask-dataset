import mmap
import os
import struct
import tempfile
from collections.abc import Buffer, Iterable
from pathlib import Path
from typing import BinaryIO

import numpy as np
import numpy.typing as npt

DEFAULT_MAX_SHARD_SIZE = 1 << 30
MAX_FIELD_SIZE = np.iinfo(np.uint32).max
INDEX_DTYPE = np.dtype("<u4")
EMPTY_VIEW = memoryview(b"")

type Entry = tuple[Buffer, Buffer]
type UIntArray = npt.NDArray[np.unsignedinteger]
type Shard = mmap.mmap | memoryview


class BitcaskDataset:
    """An append-oriented, mmap-backed collection of keyed byte blobs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

        if not self.path.exists():
            _initialize_empty_dataset(self.path)
        elif not self.path.is_dir():
            raise ValueError(f"dataset path is not a directory: {self.path}")
        elif not any(self.path.iterdir()):
            _initialize_empty_dataset(self.path)
        else:
            _validate_layout(self.path)

        self._load()

    def extend(
        self,
        entries: Iterable[Entry],
        *,
        max_shard_size: int = DEFAULT_MAX_SHARD_SIZE,
    ) -> None:
        """Add unique key/value pairs using new, immutable shards."""
        if max_shard_size <= 0:
            raise ValueError("max_shard_size must be positive")

        _extend_dataset(
            self.path,
            entries,
            existing_keys=self._keys,
            first_shard_id=len(self._shards),
            max_shard_size=max_shard_size,
        )
        self._load()

    def __len__(self) -> int:
        return self._sizes.shape[0]

    def __contains__(self, key: object) -> bool:
        try:
            normalized = _as_key(key)
        except (TypeError, ValueError):
            return False
        return normalized in self._keys

    def __getitem__(self, key: Buffer) -> memoryview:
        normalized = _as_key(key)
        try:
            index = self._keys[normalized]
        except KeyError:
            raise KeyError(normalized) from None
        return self._value_at(index)

    def at(self, index: int) -> memoryview:
        """Return a value by insertion position."""
        return self._value_at(self._normalize_index(index))

    def key_at(self, index: int) -> memoryview:
        """Return a key by insertion position."""
        return self._key_at(self._normalize_index(index))

    def __getstate__(self) -> Path:
        return self.path

    def __setstate__(self, path: Path) -> None:
        self.path = path
        self._load()

    def _load(self) -> None:
        index_path = self.path / "index.bin"
        if index_path.stat().st_size:
            raw_sizes = np.memmap(index_path, dtype=INDEX_DTYPE, mode="r")
            self._sizes = raw_sizes.reshape(-1, 2)
        else:
            self._sizes = np.empty((0, 2), dtype=INDEX_DTYPE)

        shard_paths = _shard_paths(self.path / "shards")
        shard_sizes = np.asarray([path.stat().st_size for path in shard_paths], dtype=np.uint64)
        expected_size = int(self._sizes.astype(np.uint64).sum())
        actual_size = int(shard_sizes.sum())
        if actual_size != expected_size:
            raise ValueError(f"shard data size is {actual_size}, expected {expected_size}")

        shards = [_mmap_readonly(path) for path in shard_paths]
        shard_ids, record_offsets = _build_locations(self._sizes, shard_sizes)

        self._shards = shards
        self._shard_ids = shard_ids
        self._record_offsets = record_offsets
        self._keys: dict[bytes, int] = {}
        for index in range(len(self)):
            key = bytes(self._key_at(index))
            if key in self._keys:
                raise ValueError(f"duplicate key in dataset: {key!r}")
            self._keys[key] = index

    def _key_at(self, index: int) -> memoryview:
        key_size = int(self._sizes[index, 0])
        if key_size == 0:
            return EMPTY_VIEW

        shard_id = int(self._shard_ids[index])
        offset = int(self._record_offsets[index])
        return memoryview(self._shards[shard_id])[offset : offset + key_size]

    def _value_at(self, index: int) -> memoryview:
        key_size = int(self._sizes[index, 0])
        value_size = int(self._sizes[index, 1])
        if value_size == 0:
            return EMPTY_VIEW

        shard_id = int(self._shard_ids[index])
        offset = int(self._record_offsets[index]) + key_size
        return memoryview(self._shards[shard_id])[offset : offset + value_size]

    def _normalize_index(self, index: int) -> int:
        if not isinstance(index, int):
            raise TypeError("position must be an integer")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return index


def _initialize_empty_dataset(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "shards").mkdir()
    (path / "index.bin").touch()


def _validate_layout(path: Path) -> None:
    index_path = path / "index.bin"
    shards_path = path / "shards"
    if not index_path.is_file():
        raise ValueError(f"missing dataset index: {index_path}")
    if not shards_path.is_dir():
        raise ValueError(f"missing shards directory: {shards_path}")
    if index_path.stat().st_size % 8:
        raise ValueError("index size is not a multiple of 8 bytes")


def _extend_dataset(
    path: Path,
    entries: Iterable[Entry],
    *,
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

            for key_buffer, value_buffer in entries:
                key = _as_bytes_view(key_buffer)
                value = _as_bytes_view(value_buffer)
                key_copy = bytes(key)

                if key_copy in existing_keys or key_copy in new_keys:
                    raise ValueError(f"duplicate key: {key_copy!r}")
                new_keys.add(key_copy)

                if key.nbytes > MAX_FIELD_SIZE or value.nbytes > MAX_FIELD_SIZE:
                    raise ValueError("key and value must each fit in uint32")

                record_size = key.nbytes + value.nbytes
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
                shard_file.write(value)
                shard_size += record_size
                index_file.write(struct.pack("<II", key.nbytes, value.nbytes))

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
    record_sizes = sizes[:, 0].astype(np.uint64) + sizes[:, 1]
    record_ends = np.cumsum(record_sizes, dtype=np.uint64)
    global_starts = np.empty_like(record_ends)
    if global_starts.size:
        global_starts[0] = 0
        global_starts[1:] = record_ends[:-1]

    shard_ids = np.zeros(record_sizes.size, dtype=np.uint32)
    record_offsets = np.zeros(record_sizes.size, dtype=np.uint64)
    nonempty = record_sizes != 0
    if not np.any(nonempty):
        return shard_ids, record_offsets

    shard_ends = np.cumsum(shard_sizes, dtype=np.uint64)
    if not shard_ends.size:
        raise ValueError("index references data but there are no shards")

    resolved_ids = np.searchsorted(shard_ends, global_starts[nonempty], side="right")
    if np.any(resolved_ids >= shard_ends.size):
        raise ValueError("index references data beyond the shards")
    if shard_ends.size > MAX_FIELD_SIZE:
        raise ValueError("dataset has too many shards")

    shard_starts = shard_ends - shard_sizes
    resolved_offsets = global_starts[nonempty] - shard_starts[resolved_ids]
    if np.any(resolved_offsets + record_sizes[nonempty] > shard_sizes[resolved_ids]):
        raise ValueError("an indexed record crosses a shard boundary")

    shard_ids[nonempty] = resolved_ids.astype(np.uint32)
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


def _as_key(value: object) -> bytes:
    if not isinstance(value, Buffer):
        raise TypeError("keys must support the buffer protocol")
    return bytes(_as_bytes_view(value))


def _as_bytes_view(value: Buffer) -> memoryview:
    try:
        view = memoryview(value)
    except TypeError:
        raise TypeError("keys and values must support the buffer protocol") from None
    if not view.contiguous:
        raise ValueError("keys and values must be contiguous")
    return view.cast("B")


def _copy_file(source: BinaryIO, destination: BinaryIO) -> None:
    while chunk := source.read(1024 * 1024):
        destination.write(chunk)


def _finish_file(file: BinaryIO) -> None:
    file.flush()
    os.fsync(file.fileno())
    file.close()
