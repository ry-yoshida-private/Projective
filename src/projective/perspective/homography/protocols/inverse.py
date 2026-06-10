from __future__ import annotations

from functools import cached_property
from typing import Protocol

from ....types import FloatArray


class HomographyInverseProtocol(Protocol):
    """Homography container that exposes a cached matrix inverse."""

    value: FloatArray

    @cached_property
    def inverse(self) -> FloatArray: ...
