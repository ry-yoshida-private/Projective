from __future__ import annotations

import numpy as np
from functools import cached_property
from typing import TYPE_CHECKING, Callable, cast

from ...mixin.transform import PerspectiveMatrixTransformMixin

if TYPE_CHECKING:
    from ..matrix import HomographyMatrix


class HomographyMatrixTransformMixin(PerspectiveMatrixTransformMixin):
    """Coordinate transformation operations for homography matrices."""

    value: np.ndarray

    def projective_transformation(
        self,
        points: np.ndarray,
        is_inverse: bool = False,
        up_axis_index: int = 2,
    ) -> np.ndarray:
        """
        Apply the homography transformation to 2D points.

        Parameters:
        ----------
        points: np.ndarray
            Input points of shape (N, 2) or (N, 3).
                                 - (x, y) will be automatically converted to (x, y, 1).
                                 - (x, y, w) is treated as homogeneous coordinates.
        is_inverse: bool
            If True, apply the inverse transformation to homography matrix.
        up_axis_index: int
            The index of the up axis. 0 or 1 or 2.

        Returns:
            np.ndarray: Transformed points of shape (N, 2).
        """
        if up_axis_index not in (0, 1, 2):
            raise ValueError("Up axis index must be 0, 1, or 2")
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("Input points must have shape (N, 2) or (N, 3)")

        if points.shape[1] == 2:
            ones = np.ones((points.shape[0], 1))
            if up_axis_index == 0:
                points = np.hstack([points, ones])
            elif up_axis_index == 1:
                points = np.hstack([points[:, 0], ones, points[:, 1]])
            elif up_axis_index == 2:
                points = np.hstack([points, ones])

        matrix = self.inverse if is_inverse else self.value
        transformed = (matrix @ points.T).T

        numerator_getters: dict[int, Callable[[np.ndarray], np.ndarray]] = {
            0: lambda arr: arr[:, 1:],
            1: lambda arr: arr[:, (0, 2)],
            2: lambda arr: arr[:, :2],
        }
        numerator = numerator_getters[up_axis_index](transformed)
        denominators = transformed[:, up_axis_index][:, None]
        return numerator / denominators

    @cached_property
    def inverse(self) -> np.ndarray:
        return np.linalg.inv(self.value)

    @property
    def T(self) -> np.ndarray:
        return self.value.T

    def scale_correction(self, scale: float) -> HomographyMatrix:
        """
        Correct the scale of the homography matrix.

        Parameters
        ----------
        scale : float
            Scale factor.

        Returns
        -------
        HomographyMatrix
            Homography matrix.
        """
        S = np.diag([scale, scale, 1])
        S_inv = np.linalg.inv(S)
        scaled_matrix = S_inv @ self.value @ S
        from ..matrix import HomographyMatrix as _HomographyMatrix

        return _HomographyMatrix(value=scaled_matrix)
