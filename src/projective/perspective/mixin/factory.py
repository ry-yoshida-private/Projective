from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from opencv_utility import OpenCVOutlierFilteringFlag

from ...types import FloatArray, MaskArray
from ..method import PerspectiveTransformationMethod

if TYPE_CHECKING:
    from ..matrix import PerspectiveMatrix


class PerspectiveFactoryMixin(ABC):
    """Factory methods shared by all perspective matrix types."""

    @staticmethod
    def _validate_points(
        origin_points: FloatArray,
        destination_points: FloatArray,
    ) -> bool:
        """
        Validate input points for matrix creation.

        Parameters
        ----------
        origin_points: FloatArray
            Origin points (n, 2).
        destination_points: FloatArray
            Destination points (n, 2).

        Returns
        -------
            bool: True if points are valid, False otherwise
        """
        if len(origin_points) == 0 or len(destination_points) == 0:
            return False
        if origin_points.shape != destination_points.shape:
            return False
        if origin_points.shape[1] != 2 or destination_points.shape[1] != 2:
            return False
        if len(origin_points) < 4 or len(destination_points) < 4:
            return False
        return True

    @classmethod
    def from_points(
        cls,
        origin_points: FloatArray,
        destination_points: FloatArray,
        transform_type: PerspectiveTransformationMethod = PerspectiveTransformationMethod.HOMOGRAPHY,
        outlier_filtering_flag: OpenCVOutlierFilteringFlag = OpenCVOutlierFilteringFlag.RANSAC,
        ransac_th: float = 3.0,
    ) -> tuple[PerspectiveMatrix, MaskArray]:
        """
        Estimate a perspective matrix from origin and destination point pairs.

        Parameters
        ----------
        origin_points : FloatArray
            Origin points (n, 2).
        destination_points : FloatArray
            Destination points (n, 2).
        transform_type : PerspectiveTransformationMethod
            Affine or homography estimation.
        outlier_filtering_flag : OpenCVOutlierFilteringFlag
            Outlier filtering method passed to OpenCV.
        ransac_th : float
            RANSAC reprojection threshold.

        Returns
        -------
        tuple[PerspectiveMatrix, MaskArray]
            The estimated perspective matrix and inlier mask of shape (N, 1).
        """
        matrix, mask = transform_type.perspective_class.create_from_points(
            origin_points,
            destination_points,
            outlier_filtering_flag,
            ransac_th,
        )
        return matrix, mask

    @classmethod
    @abstractmethod
    def create_identity_matrix(cls) -> PerspectiveMatrix:
        """
        Create an identity perspective matrix.

        Returns
        -------
        PerspectiveMatrix:
            The identity perspective matrix.
        """

    @classmethod
    @abstractmethod
    def create_from_points(
        cls,
        origin_points: FloatArray,
        destination_points: FloatArray,
        outlier_filtering_flag: OpenCVOutlierFilteringFlag = OpenCVOutlierFilteringFlag.RANSAC,
        ransac_th: float = 3.0,
    ) -> tuple[PerspectiveMatrix, MaskArray]:
        """
        Create a perspective matrix container from a set of origin and destination points.

        Parameters
        ----------
        origin_points : FloatArray
            Origin points (n, 2).
        destination_points : FloatArray
            Destination points (n, 2).
        outlier_filtering_flag : OpenCVOutlierFilteringFlag
            Outlier filtering method passed to OpenCV.
        ransac_th : float
            RANSAC reprojection threshold.

        Returns
        -------
        tuple[PerspectiveMatrix, MaskArray]
            The perspective matrix and inlier mask of shape (N, 1).
        """
