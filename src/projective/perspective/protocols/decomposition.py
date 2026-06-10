from __future__ import annotations

from typing import Protocol

from ...types import FloatArray
from ..method import PerspectiveTransformationMethod


class PerspectiveMatrixDecompositionProtocol(Protocol):
    """Protocol for decomposition properties of a perspective matrix."""

    value: FloatArray

    @property
    def translation(self) -> FloatArray: ...

    @property
    def shear(self) -> FloatArray: ...

    @property
    def has_perspective(self) -> bool: ...

    @property
    def transform_type(self) -> PerspectiveTransformationMethod: ...
