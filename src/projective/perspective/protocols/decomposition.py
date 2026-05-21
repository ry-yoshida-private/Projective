from __future__ import annotations

from typing import Protocol

import numpy as np

from ..method import PerspectiveTransformationMethod


class PerspectiveMatrixDecompositionProtocol(Protocol):
    """Protocol for decomposition properties of a perspective matrix."""

    value: np.ndarray

    @property
    def translation(self) -> np.ndarray: ...

    @property
    def shear(self) -> np.ndarray: ...

    @property
    def has_perspective(self) -> bool: ...

    @property
    def transform_type(self) -> PerspectiveTransformationMethod: ...
