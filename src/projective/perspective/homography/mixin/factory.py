from __future__ import annotations

import cv2
import numpy as np
from typing import TYPE_CHECKING, cast

from opencv_utility import OpenCVOutlierFilteringFlag

from ...mixin.factory import PerspectiveFactoryMixin

if TYPE_CHECKING:
    from ..matrix import HomographyMatrix


class HomographyFactoryMixin(PerspectiveFactoryMixin):
    """Factory methods for homography matrix estimation."""

    @classmethod
    def create_identity_matrix(cls) -> HomographyMatrix:
        """
        Create an identity homography matrix.

        Returns
        -------
        HomographyMatrix
            Identity homography matrix.
        """
        from ..matrix import HomographyMatrix as _HomographyMatrix

        return _HomographyMatrix(value=np.eye(3, 3))

    @classmethod
    def from_unnormalized_value(
        cls,
        value: np.ndarray,
    ) -> HomographyMatrix:
        """
        Create a homography matrix from an unnormalized value.

        Parameters
        ----------
        value: np.ndarray
            Unnormalized value of the homography matrix (shape: (3, 3)).

        Returns
        -------
        HomographyMatrix:
        """
        if value.shape != (3, 3):
            raise ValueError("Unnormalized value must be a 3x3 matrix")
        from ..matrix import HomographyMatrix as _HomographyMatrix

        return _HomographyMatrix(value=value / value[2, 2])

    @classmethod
    def create_from_points(
        cls,
        origin_points: np.ndarray,
        destination_points: np.ndarray,
        outlier_filtering_flag: OpenCVOutlierFilteringFlag = OpenCVOutlierFilteringFlag.RANSAC,
        ransac_th: float = 3.0,
    ) -> tuple[HomographyMatrix, np.ndarray]:
        """
        Create a homography matrix from a set of origin and destination points.

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
        tuple[HomographyMatrix, np.ndarray]
            Homography matrix and inlier mask of shape (N, 1).
        """
        if not cls._validate_points(origin_points, destination_points):
            raise ValueError("Invalid points for homography matrix creation")

        origin_points = np.asarray(origin_points, dtype=np.float32)
        destination_points = np.asarray(destination_points, dtype=np.float32)

        matrix, mask = cast(
            tuple[np.ndarray, np.ndarray],
            cv2.findHomography(
                srcPoints=origin_points,
                dstPoints=destination_points,
                method=outlier_filtering_flag.cv2_flag,
                ransacReprojThreshold=ransac_th,
            ),
        )
        from ..matrix import HomographyMatrix as _HomographyMatrix

        return _HomographyMatrix(value=matrix), mask
