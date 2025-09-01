import torch
from geo_seg.data.random import RandomTileDataset


def test_random_dataset_length_and_shapes():
    ds = RandomTileDataset()
    assert len(ds) == 128
    sample = ds[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["mask"].shape == (64, 64)


def test_random_dataset_determinism():
    ds1 = RandomTileDataset(seed=123)
    ds2 = RandomTileDataset(seed=123)
    s1, s2 = ds1[10], ds2[10]
    assert torch.allclose(s1["image"], s2["image"])  # deterministic
    assert torch.equal(s1["mask"], s2["mask"])  # exact equality for ints
