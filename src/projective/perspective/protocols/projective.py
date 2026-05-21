from __future__ import annotations

from typing import Protocol

import numpy as np


class PerspectiveProjectiveProtocol(Protocol):
    """Protocol for projecting 2D points with a perspective matrix."""

    value: np.ndarray

    def projective_transformation(
        self,
        points: np.ndarray,
        is_inverse: bool = False,
        up_axis_index: int = 2,
    ) -> np.ndarray: ...
