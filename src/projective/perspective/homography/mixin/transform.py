from __future__ import annotations

import numpy as np
from functools import cached_property
from typing import TYPE_CHECKING

from ...mixin.transform import PerspectiveTransformMixin

if TYPE_CHECKING:
    from ..matrix import HomographyMatrix


class HomographyTransformMixin(PerspectiveTransformMixin):
    """Transform the homography matrix container itself."""

    @property
    def column_vector(self) -> np.ndarray:
        """
        Return the column vector of a 3×3 homography matrix.

        Returns
        -------
        np.ndarray
            Column vector with shape (9, 1).
        """
        return super().column_vector

    @property
    def row_vector(self) -> np.ndarray:
        """
        Return the row vector of a 3×3 homography matrix.

        Returns
        -------
        np.ndarray
            Row vector with shape (1, 9).
        """
        return super().row_vector

    @property
    def shape(self) -> tuple[int, int]:
        """
        Return the shape of the homography matrix.

        Returns
        -------
        tuple[int, int]
            Matrix shape (3, 3).
        """
        return super().shape

    @property
    def flatten(self) -> np.ndarray:
        """
        Return the flattened homography matrix.

        Returns
        -------
        np.ndarray
            One-dimensional array with shape (9,).
        """
        return super().flatten

    @cached_property
    def inverse(self) -> np.ndarray:
        """
        Return the inverse of the homography matrix.

        Returns
        -------
        np.ndarray
            Inverse matrix with shape (3, 3).
        """
        return np.linalg.inv(self.value)

    @property
    def T(self) -> np.ndarray:
        """
        Return the transpose of the homography matrix.

        Returns
        -------
        np.ndarray
            Transposed matrix with shape (3, 3).
        """
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
            Homography matrix with scale corrected under similarity conjugation.
        """
        S = np.diag([scale, scale, 1])
        S_inv = np.linalg.inv(S)
        scaled_matrix = S_inv @ self.value @ S
        from ..matrix import HomographyMatrix as _HomographyMatrix

        return _HomographyMatrix(value=scaled_matrix)
