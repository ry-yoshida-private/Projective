from __future__ import annotations

from typing import Protocol

from ....types import FloatArray


class HomographyMatrixValueProtocol(Protocol):
    """Protocol for homography matrix instances."""

    value: FloatArray
