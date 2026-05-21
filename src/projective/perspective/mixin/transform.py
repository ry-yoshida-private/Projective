from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..matrix import PerspectiveMatrix


class PerspectiveMatrixTransformMixin(ABC):
    """Coordinate transformation operations for perspective matrices."""

    value: np.ndarray

    @abstractmethod
    def scale_correction(
        self,
        scale: float,
    ) -> PerspectiveMatrix:
        """
        Correct the scale of the perspective matrix.

        Parameters
        ----------
        scale: float
            Scale factor.

        Returns
        -------
        PerspectiveMatrix:
            The perspective matrix with corrected scale.
        """

    @abstractmethod
    def projective_transformation(
        self,
        points: np.ndarray,
        is_inverse: bool = False,
        up_axis_index: int = 2,
    ) -> np.ndarray:
        """
        Apply the perspective transformation to 2D points.

        Parameters:
        ----------
        points: np.ndarray
            Input points of shape (N, 2) or (N, 3).
        is_inverse: bool
            If True, apply the inverse transformation to perspective matrix.
        up_axis_index: int
            Homography only: index of the up axis (0, 1, or 2). Ignored by affine matrices.

        Returns:
            np.ndarray: Transformed points of shape (N, 2).
        """
