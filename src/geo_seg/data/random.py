import torch
from torch.utils.data import Dataset
from .base import Sample


class RandomTileDataset(Dataset[Sample]):
    """Tiny deterministic random dataset for CPU sanity checks.

    - Images: shape [3, 64, 64], values in [0, 1].
    - Masks: shape [64, 64], binary {0,1} with simple threshold.
    - Length: 128 by default.
    - Deterministic given a seed.
    """

    def __init__(self, length: int = 128, seed: int = 123, image_size: int = 64):
        self.length = int(length)
        self.image_size = int(image_size)
        self.seed = int(seed)

        # Pre-generate tensors deterministically for fast access and strict determinism
        g = torch.Generator()
        g.manual_seed(self.seed)
        imgs = torch.rand(
            (self.length, 3, self.image_size, self.image_size), generator=g
        )
        # simple synthetic mask: foreground where channel 0 exceeds 0.5
        masks = (imgs[:, 0] > 0.5).to(torch.long)
        self._images = imgs
        self._masks = masks

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Sample:
        img = self._images[idx]
        mask = self._masks[idx]
        return {"image": img, "mask": mask}
