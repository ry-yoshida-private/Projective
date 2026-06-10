from __future__ import annotations

from typing import Protocol

from ...types import FloatArray


class MatrixValueProtocol(Protocol):
    """Protocol for matrix containers that expose a ``value`` ndarray."""

    value: FloatArray
