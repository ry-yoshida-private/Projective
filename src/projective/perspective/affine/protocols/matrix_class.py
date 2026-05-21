from __future__ import annotations

from typing import Protocol

import numpy as np

from .matrix_value import AffineMatrixValueProtocol


class AffineMatrixClassProtocol(Protocol):
    """Protocol for the ``AffineMatrix`` class object used by factory mixins."""

    @staticmethod
    def _validate_points(
        origin_points: np.ndarray,
        destination_points: np.ndarray,
    ) -> bool: ...

    def __call__(self, *, value: np.ndarray) -> AffineMatrixValueProtocol: ...
