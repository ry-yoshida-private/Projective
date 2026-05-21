from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from ...mixin.transform import PerspectiveMatrixTransformMixin

if TYPE_CHECKING:
    from ..matrix import AffineMatrix


class AffineMatrixTransformMixin(PerspectiveMatrixTransformMixin):
    """Coordinate transformation operations for affine matrices."""

    value: np.ndarray

    def projective_transformation(
        self,
        points: np.ndarray,
        is_inverse: bool = False,
        up_axis_index: int = 2,
    ) -> np.ndarray:
        """
        Apply the affine transformation to 2D points.

        Parameters:
        ----------
        points: np.ndarray
            Input points of shape (N, 2) or (N, 3).
                                 - (x, y) will be automatically converted to (x, y, 1).
                                 - (x, y, w) is treated as homogeneous coordinates.

        Returns:
            np.ndarray: Transformed points of shape (N, 2).
        """
        if is_inverse:
            raise ValueError("Inverse transformation is not supported for affine matrix")
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("Input points must have shape (N, 2) or (N, 3)")

        if points.shape[1] == 2:
            ones = np.ones((points.shape[0], 1))
            points_3d = np.hstack([points, ones])
        else:
            points_3d = points

        transformed = (self.value @ points_3d.T).T

        if points.shape[1] == 3:
            w = points_3d[:, 2]
            transformed = transformed / w[:, np.newaxis]

        return transformed

    def scale_correction(
        self,
        scale: float,
    ) -> AffineMatrix:
        """
        Correct the scale of the affine matrix.

        Parameters
        ----------
        scale : float
            Scale factor.

        Returns
        -------
        AffineMatrix
            Affine matrix.
        """
        extended = np.vstack([self.value, [0, 0, 1]])

        S = np.diag([scale, scale, 1])
        S_inv = np.diag([1 / scale, 1 / scale, 1])

        corrected = S_inv @ extended @ S
        from ..matrix import AffineMatrix as _AffineMatrix

        return _AffineMatrix(value=corrected[:2])
