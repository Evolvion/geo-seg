from typing import Protocol, TypedDict
from torch import Tensor


class Sample(TypedDict):
    image: Tensor  # shape [C, H, W]
    mask: Tensor  # shape [H, W]


class DatasetProtocol(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> Sample: ...
