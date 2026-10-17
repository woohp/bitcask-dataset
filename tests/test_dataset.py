import mmap
import pickle
from pathlib import Path

import numpy as np
import pytest

from bitcask_dataset import BitcaskDataset


def test_create_extend_and_lookup(tmp_path: Path) -> None:
    dataset = BitcaskDataset.create(tmp_path / "data", keyed=True)
    dataset.extend([(b"a", b"alpha"), (b"b", b"beta")])

    assert len(dataset) == 2
    assert bytes(dataset[b"a"]) == b"alpha"
    assert bytes(dataset.at(1)) == b"beta"
    assert bytes(dataset.at(-1)) == b"beta"
    assert bytes(dataset.at(np.int64(1))) == b"beta"
    assert b"a" in dataset
    assert b"missing" not in dataset


def test_reopen_reads_settings_from_manifest(tmp_path: Path) -> None:
    path = tmp_path / "data"
    BitcaskDataset.create(path, keyed=True, compression=None).extend([
        (b"key", b"value")
    ])

    reopened = BitcaskDataset(path)
    assert reopened.keyed is True
    assert reopened.compression is None
    assert bytes(reopened[b"key"]) == b"value"
    assert bytes(reopened.key_at(0)) == b"key"


def test_pickle_reopens_dataset(tmp_path: Path) -> None:
    dataset = BitcaskDataset.create(tmp_path / "data", keyed=True)
    dataset.extend([(b"key", b"value")])

    reopened = pickle.loads(pickle.dumps(dataset))

    assert reopened.path == dataset.path
    assert bytes(reopened[b"key"]) == b"value"


def test_keyless_dataset(tmp_path: Path) -> None:
    path = tmp_path / "data"
    dataset = BitcaskDataset.create(path)
    dataset.extend([b"alpha", b"beta", b""])

    assert len(dataset) == 3
    assert bytes(dataset.at(0)) == b"alpha"
    assert bytes(dataset.at(-1)) == b""
    assert b"alpha" not in dataset
    with pytest.raises(TypeError, match="key lookup is disabled"):
        dataset[b"alpha"]
    with pytest.raises(TypeError, match="keys are disabled"):
        dataset.key_at(0)

    reopened = BitcaskDataset(path)
    assert bytes(reopened.at(1)) == b"beta"


def test_extend_uses_new_shards(tmp_path: Path) -> None:
    dataset = BitcaskDataset.create(tmp_path / "data", keyed=True, compression=None)
    dataset.extend([(b"a", b"1")], max_shard_size=10)
    dataset.extend([(b"b", b"2")], max_shard_size=10)

    assert sorted(path.name for path in dataset.path.glob("0*")) == [
        "000000.bin",
        "000000.idx",
        "000001.bin",
        "000001.idx",
    ]
    assert bytes(dataset[b"a"]) == b"1"
    assert bytes(dataset[b"b"]) == b"2"


def test_shard_boundaries_and_oversized_record(tmp_path: Path) -> None:
    dataset = BitcaskDataset.create(tmp_path / "data", keyed=True, compression=None)
    dataset.extend(
        [(b"a", b"12"), (b"b", b"34"), (b"large", b"x" * 20)],
        max_shard_size=6,
    )

    assert len(list(dataset.path.glob("*.bin"))) == 2
    assert bytes(dataset[b"large"]) == b"x" * 20


def test_empty_key_and_value(tmp_path: Path) -> None:
    dataset = BitcaskDataset.create(tmp_path / "data", keyed=True)
    dataset.extend([(b"", b""), (b"key", b"")])

    assert bytes(dataset[b""]) == b""
    assert bytes(dataset[b"key"]) == b""


def test_duplicate_keys_are_rejected_without_commit(tmp_path: Path) -> None:
    dataset = BitcaskDataset.create(tmp_path / "data", keyed=True)
    dataset.extend([(b"a", b"old")])

    with pytest.raises(ValueError, match="duplicate key"):
        dataset.extend([(b"b", b"new"), (b"a", b"replacement")])

    reopened = BitcaskDataset(dataset.path)
    assert len(reopened) == 1
    assert bytes(reopened[b"a"]) == b"old"
    assert b"b" not in reopened
    assert not (dataset.path / "000001.bin").exists()


def test_uncommitted_shard_is_ignored_and_replaced(tmp_path: Path) -> None:
    dataset = BitcaskDataset.create(tmp_path / "data", keyed=True, compression=None)
    dataset.extend([(b"a", b"1")])
    (dataset.path / "000001.bin").write_bytes(b"orphan")
    (dataset.path / "000001.idx").write_bytes(b"junk")

    reopened = BitcaskDataset(dataset.path)
    assert len(reopened) == 1
    reopened.extend([(b"b", b"2")])

    assert bytes(BitcaskDataset(dataset.path)[b"b"]) == b"2"


def test_stale_handle_does_not_overwrite_committed_shards(tmp_path: Path) -> None:
    path = tmp_path / "data"
    BitcaskDataset.create(path, keyed=True)
    first, second = BitcaskDataset(path), BitcaskDataset(path)
    first.extend([(b"a", b"1")])
    second.extend([(b"b", b"2")])

    reopened = BitcaskDataset(path)
    assert bytes(reopened[b"a"]) == b"1"
    assert bytes(reopened[b"b"]) == b"2"


def test_releasing_an_empty_value_does_not_affect_later_reads(tmp_path: Path) -> None:
    dataset = BitcaskDataset.create(tmp_path / "data")
    dataset.extend([b"", b""])

    with dataset.at(0):
        pass

    assert bytes(dataset.at(1)) == b""
    dataset.extend([b"value"])
    assert bytes(dataset.at(2)) == b"value"


def test_create_requires_empty_directory(tmp_path: Path) -> None:
    path = tmp_path / "data"
    path.mkdir()
    (path / "other").write_text("not a dataset")

    with pytest.raises(FileExistsError):
        BitcaskDataset.create(path)
    with pytest.raises(FileNotFoundError):
        BitcaskDataset(path)


def test_shard_not_matching_its_index(tmp_path: Path) -> None:
    dataset = BitcaskDataset.create(tmp_path / "data", compression=None)
    dataset.extend([b"value"])
    (dataset.path / "000000.bin").write_bytes(b"val")

    with pytest.raises(ValueError, match="does not match its index"):
        BitcaskDataset(dataset.path)


def test_index_contains_uint32_sizes(tmp_path: Path) -> None:
    dataset = BitcaskDataset.create(tmp_path / "data", keyed=True, compression=None)
    dataset.extend([(b"abc", b"value")])

    sizes = np.fromfile(dataset.path / "000000.idx", dtype="<u4").reshape(-1, 2)
    np.testing.assert_array_equal(sizes, [[3, 5]])


def test_uncompressed_values_are_zero_copy(tmp_path: Path) -> None:
    dataset = BitcaskDataset.create(tmp_path / "data", compression=None)
    dataset.extend([b"value"])

    value = dataset.at(0)
    assert isinstance(value.obj, mmap.mmap)
    assert value.readonly
    assert bytes(value) == b"value"


def test_lz4_compresses_in_order(tmp_path: Path) -> None:
    dataset = BitcaskDataset.create(tmp_path / "data")
    values = [bytes([i]) * 10_000 for i in range(100)]
    dataset.extend(values)

    stored_size = int(np.fromfile(dataset.path / "000000.idx", dtype="<u4")[0])
    assert stored_size < len(values[0])
    assert [bytes(dataset.at(i)) for i in range(len(values))] == values


def test_invalid_compression(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported compression"):
        BitcaskDataset.create(tmp_path / "data", compression="gzip")  # type: ignore[arg-type]


def test_missing_key_and_bad_position(tmp_path: Path) -> None:
    dataset = BitcaskDataset.create(tmp_path / "data", keyed=True)

    with pytest.raises(KeyError):
        dataset[b"missing"]
    with pytest.raises(IndexError):
        dataset.at(0)
