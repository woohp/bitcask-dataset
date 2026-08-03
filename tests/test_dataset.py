import pickle
from pathlib import Path

import numpy as np
import pytest

from bitcask_dataset import BitcaskDataset


def test_create_extend_and_lookup(tmp_path: Path) -> None:
    dataset = BitcaskDataset(tmp_path / "data", keyed=True)
    dataset.extend([(b"a", b"alpha"), (b"b", b"beta")])

    assert len(dataset) == 2
    assert bytes(dataset[b"a"]) == b"alpha"
    assert bytes(dataset.at(1)) == b"beta"
    assert bytes(dataset.at(-1)) == b"beta"
    assert b"a" in dataset
    assert b"missing" not in dataset


def test_reopen(tmp_path: Path) -> None:
    path = tmp_path / "data"
    BitcaskDataset(path, keyed=True).extend([(b"key", b"value")])

    reopened = BitcaskDataset(path, keyed=True)
    assert bytes(reopened[b"key"]) == b"value"
    assert bytes(reopened.key_at(0)) == b"key"


def test_pickle_reopens_dataset(tmp_path: Path) -> None:
    dataset = BitcaskDataset(tmp_path / "data", keyed=True)
    dataset.extend([(b"key", b"value")])

    reopened = pickle.loads(pickle.dumps(dataset))

    assert reopened.path == dataset.path
    assert bytes(reopened[b"key"]) == b"value"


def test_keyless_dataset(tmp_path: Path) -> None:
    path = tmp_path / "data"
    dataset = BitcaskDataset(path, keyed=False)
    dataset.extend([b"alpha", b"beta", b""])

    assert len(dataset) == 3
    assert bytes(dataset.at(0)) == b"alpha"
    assert bytes(dataset.at(-1)) == b""
    assert b"alpha" not in dataset
    with pytest.raises(TypeError, match="key lookup is disabled"):
        dataset[b"alpha"]
    with pytest.raises(TypeError, match="keys are disabled"):
        dataset.key_at(0)

    reopened = BitcaskDataset(path, keyed=False)
    assert bytes(reopened.at(1)) == b"beta"
    assert pickle.loads(pickle.dumps(reopened)).keyed is False

    sizes = np.fromfile(path / "index.bin", dtype="<u4")
    np.testing.assert_array_equal(sizes, [5, 4, 0])


def test_extend_uses_new_shards(tmp_path: Path) -> None:
    dataset = BitcaskDataset(tmp_path / "data", keyed=True)
    dataset.extend([(b"a", b"1")], max_shard_size=10)
    dataset.extend([(b"b", b"2")], max_shard_size=10)

    assert sorted(path.name for path in (dataset.path / "shards").glob("*.bin")) == [
        "000000.bin",
        "000001.bin",
    ]
    assert bytes(dataset[b"a"]) == b"1"
    assert bytes(dataset[b"b"]) == b"2"


def test_shard_boundaries_and_oversized_record(tmp_path: Path) -> None:
    dataset = BitcaskDataset(tmp_path / "data", keyed=True)
    dataset.extend(
        [(b"a", b"12"), (b"b", b"34"), (b"large", b"x" * 20)],
        max_shard_size=6,
    )

    assert len(list((dataset.path / "shards").glob("*.bin"))) == 2
    assert bytes(dataset[b"large"]) == b"x" * 20


def test_empty_key_and_value(tmp_path: Path) -> None:
    dataset = BitcaskDataset(tmp_path / "data", keyed=True)
    dataset.extend([(b"", b""), (b"key", b"")])

    assert bytes(dataset[b""]) == b""
    assert bytes(dataset[b"key"]) == b""


def test_duplicate_keys_are_rejected_without_commit(tmp_path: Path) -> None:
    dataset = BitcaskDataset(tmp_path / "data", keyed=True)
    dataset.extend([(b"a", b"old")])

    with pytest.raises(ValueError, match="duplicate key"):
        dataset.extend([(b"b", b"new"), (b"a", b"replacement")])

    reopened = BitcaskDataset(dataset.path, keyed=True)
    assert len(reopened) == 1
    assert bytes(reopened[b"a"]) == b"old"
    assert b"b" not in reopened


def test_invalid_existing_directory(tmp_path: Path) -> None:
    path = tmp_path / "data"
    path.mkdir()
    (path / "other").write_text("not a dataset")

    with pytest.raises(ValueError, match="missing dataset index"):
        BitcaskDataset(path, keyed=True)


def test_malformed_index(tmp_path: Path) -> None:
    path = tmp_path / "data"
    (path / "shards").mkdir(parents=True)
    (path / "index.bin").write_bytes(b"bad")

    with pytest.raises(ValueError, match="multiple of 8"):
        BitcaskDataset(path, keyed=True)


def test_index_contains_uint32_sizes(tmp_path: Path) -> None:
    dataset = BitcaskDataset(tmp_path / "data", keyed=True)
    dataset.extend([(b"abc", b"value")])

    sizes = np.fromfile(dataset.path / "index.bin", dtype="<u4").reshape(-1, 2)
    np.testing.assert_array_equal(sizes, [[3, 5]])


def test_missing_key_and_bad_position(tmp_path: Path) -> None:
    dataset = BitcaskDataset(tmp_path / "data", keyed=True)

    with pytest.raises(KeyError):
        dataset[b"missing"]
    with pytest.raises(IndexError):
        dataset.at(0)
