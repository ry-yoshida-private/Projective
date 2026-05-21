from __future__ import annotations

from typing import Protocol

import numpy as np

from .matrix_value import MatrixValueProtocol


class PerspectiveMatrixTransformProtocol(Protocol):
    """Protocol for perspective coordinate transformation operations."""

    value: np.ndarray

    def projective_transformation(
        self,
        points: np.ndarray,
        is_inverse: bool = False,
        up_axis_index: int = 2,
    ) -> np.ndarray: ...

    def scale_correction(self, scale: float) -> MatrixValueProtocol: ...
