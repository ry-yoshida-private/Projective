from __future__ import annotations

import cv2
import numpy as np
from typing import TYPE_CHECKING, cast

from opencv_utility import OpenCVOutlierFilteringFlag

from ...mixin.factory import PerspectiveFactoryMixin

if TYPE_CHECKING:
    from ..matrix import AffineMatrix


class AffineFactoryMixin(PerspectiveFactoryMixin):
    """Factory methods for affine matrix estimation."""

    @classmethod
    def create_identity_matrix(cls) -> AffineMatrix:
        """
        Create an identity affine matrix.

        Returns
        -------
        AffineMatrix:
            The identity affine matrix.
        """
        from ..matrix import AffineMatrix as _AffineMatrix

        return _AffineMatrix(value=np.eye(2, 3))

    @classmethod
    def create_from_points(
        cls,
        origin_points: np.ndarray,
        destination_points: np.ndarray,
        outlier_filtering_flag: OpenCVOutlierFilteringFlag = OpenCVOutlierFilteringFlag.RANSAC,
        ransac_th: float = 3.0,
    ) -> tuple[AffineMatrix, np.ndarray]:
        """
        Create a partial affine matrix (``cv2.estimateAffinePartial2D``) from point correspondences.

        Parameters
        ----------
        origin_points : np.ndarray
            Origin points (n, 2).
        destination_points : np.ndarray
            Destination points (n, 2).
        outlier_filtering_flag: OpenCVOutlierFilteringFlag
            Outlier filtering flag.
        ransac_th : float
            RANSAC threshold.

        Returns
        -------
        tuple[AffineMatrix, np.ndarray]
            Affine matrix and inlier mask of shape (N, 1).
        """
        if not cls._validate_points(origin_points, destination_points):
            raise ValueError("Invalid points for affine matrix creation")

        origin_points = np.asarray(origin_points, dtype=np.float32)
        destination_points = np.asarray(destination_points, dtype=np.float32)

        matrix, mask = cast(
            tuple[np.ndarray, np.ndarray],
            cv2.estimateAffinePartial2D(
                from_=origin_points,
                to=destination_points,
                method=outlier_filtering_flag.cv2_flag,
                ransacReprojThreshold=ransac_th,
            ),
        )
        from ..matrix import AffineMatrix as _AffineMatrix

        return _AffineMatrix(value=matrix), mask
