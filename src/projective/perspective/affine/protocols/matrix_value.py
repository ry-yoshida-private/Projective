from __future__ import annotations

from typing import Protocol

from ....types import FloatArray


class AffineMatrixValueProtocol(Protocol):
    """Protocol for affine matrix instances."""

    value: FloatArray
