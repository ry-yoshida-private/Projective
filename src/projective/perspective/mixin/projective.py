from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class PerspectiveProjectiveMixin(ABC):
    """Project external 2D points with the matrix (homogeneous map + dehomogenize)."""

    value: np.ndarray

    @abstractmethod
    def projective_transformation(
        self,
        points: np.ndarray,
        is_inverse: bool = False,
        up_axis_index: int = 2,
    ) -> np.ndarray:
        """
        Apply the perspective matrix to 2D points.

        Parameters
        ----------
        points : np.ndarray
            Input points of shape (N, 2) or (N, 3).
        is_inverse : bool
            If True, apply the inverse matrix.
        up_axis_index : int
            Homography only: index of the homogeneous coordinate used as denominator.

        Returns
        -------
        np.ndarray
            Projected points of shape (N, 2).
        """
