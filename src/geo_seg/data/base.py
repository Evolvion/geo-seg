from typing import Protocol, TypedDict
from torch import Tensor


class Sample(TypedDict):
    """Dataset sample spec.

    - image: Tensor[3, H, W]
    - mask:  Tensor[H, W] (integer class IDs)
    """

    image: Tensor
    mask: Tensor


class DatasetProtocol(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> Sample: ...

