from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass

from .perspective_matrix import PerspectiveMatrix
from .method import PerspectiveTransformationMethod

@dataclass
class AffineMatrix(PerspectiveMatrix):
    """
    AffineMatrix is a class that represents an affine transformation matrix.

    Attributes:
    ----------
    value: np.ndarray
        The affine transformation matrix with shape (2, 3).
    """
    
    def __post_init__(self) -> None:
        """
        Post-init validation.

        Raises
        ------
        ValueError: If the affine transformation matrix is not a 2x3 matrix.
        """
        if self.value.shape != (2, 3):
            raise ValueError(f"Affine matrix must be a 2x3 matrix, got shape {self.value.shape}")

    def projective_transformation(
        self, 
        points: np.ndarray, 
        is_inverse: bool = False
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

        return transformed
    
    @classmethod
    def create_identity_matrix(cls) -> AffineMatrix:
        """
        Create an identity affine matrix.

        Returns
        -------
        AffineMatrix:
            The identity affine matrix.
        """
        return AffineMatrix(value=np.eye(2, 3))

    def scale_correction(
        self, 
        scale: float
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
        S_inv = np.diag([1/scale, 1/scale, 1])

        corrected = S_inv @ extended @ S
        return AffineMatrix(value=corrected[:2])

    @property
    def translation(self) -> np.ndarray:
        """
        Return the translation vector (tx, ty).
        
        Returns
        -------
        np.ndarray:
            The translation vector (tx, ty) with shape (2,).
        """
        return self.value[:, 2]

    @property
    def shear(self) -> np.ndarray:
        """
        Return shear factors along x and y axes.

        Calculated from the linear part of the affine matrix:
        [a b tx]
        [c d ty]

        Returns
        -------
        np.ndarray:
            The shear factors (sx, sy) with shape (2,).
        """
        a, b = self.value[0, :2]
        c, d = self.value[1, :2]
        sx2 = a**2 + c**2
        sy2 = b**2 + d**2

        shear_x = (a*b + c*d) / sx2 if not sx2 == 0 else 0
        shear_y = (a*b + c*d) / sy2 if not sy2 == 0 else 0
        return np.array([shear_x, shear_y])

    @property
    def has_perspective(self) -> bool:
        """
        Return True if the affine matrix has a perspective component.
        
        Returns
        -------
        bool:
            True if the affine matrix has a perspective component, False otherwise.
        """
        return False

    @property
    def transform_type(self) -> PerspectiveTransformationMethod:
        return PerspectiveTransformationMethod.AFFINE

    @classmethod
    def create_from_points(
        cls,
        origin_points: np.ndarray, 
        destination_points: np.ndarray, 
        ransac_th: float = 3.0
        ) -> AffineMatrix:
        """
        Create an affine matrix from a set of origin and destination points.
        
        Parameters
        ----------
        origin_points : np.ndarray
            Origin points in homogeneous coordinates (n, 2).
        destination_points : np.ndarray
            Destination points in homogeneous coordinates (n, 2).
        ransac_th : float
            RANSAC threshold.

        Returns
        -------
        AffineMatrix
            Affine matrix.
        """
        if not cls._validate_points(origin_points, destination_points):
            return cls.create_identity_matrix()
        
        origin_points = np.asarray(origin_points, dtype=np.float32)
        destination_points = np.asarray(destination_points, dtype=np.float32)
        
        matrix, _ = cv2.estimateAffinePartial2D(
                    from_=origin_points, 
                    to=destination_points, 
                    method=cv2.RANSAC, 
                    ransacReprojThreshold=ransac_th
                    )
        return cls(value=matrix)
