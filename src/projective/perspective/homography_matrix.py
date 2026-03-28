from __future__ import annotations
import cv2
import numpy as np
import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import Callable
from .perspective_matrix import PerspectiveMatrix
from .method import PerspectiveTransformationMethod

@dataclass
class HomographyMatrix(PerspectiveMatrix):
    """
    Homography matrix H representing a 2D projective transformation.
    
    The 3x3 homography matrix H has the form:
        [a  b  tx]
    H = [c  d  ty]
        [p1 p2 1 ]
    
    Where:
    - a, b, c, d: linear transformation components (rotation, scale, shear)
    - tx, ty: translation components
    - p1, p2: perspective transformation components

    Attributes:
    ----------
    value: np.ndarray (inherited from base)
        Value of the homography matrix with shape (3, 3).
    """
    def __post_init__(self) -> None:
        if self.value.shape != (3, 3):
            raise ValueError("Homography matrix must be a 3x3 matrix")

    def projective_transformation(
        self, 
        points: np.ndarray, 
        is_inverse: bool = False,
        up_axis_index: int = 2
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
        return HomographyMatrix(value=scaled_matrix)

    @classmethod
    def create_identity_matrix(cls) -> HomographyMatrix:
        """
        Create an identity homography matrix.
        
        Returns
        -------
        HomographyMatrix
            Identity homography matrix.
        """
        return HomographyMatrix(value=np.eye(3, 3))

    @property
    def translation(self) -> np.ndarray:
        """
        Approximate translation vector (tx, ty) from homography.
        
        Extracts the translation components from the third column of H:
        H[0,2] = tx, H[1,2] = ty
        """
        return self.value[:2, 2]

    @property
    def shear(self) -> np.ndarray:
        """
        Return the shear components of the homography matrix.
        
        Returns:
        ----------
        np.ndarray:
            The shear of the homography matrix.
        """
        H_norm = self.value / self.value[2, 2]
        A = H_norm[:2, :2]
        _, S = np.linalg.qr(A) # R, S: np.ndarray

        sx = np.linalg.norm(S[:, 0])
        sy = np.linalg.norm(S[:, 1])

        shear_x = S[0, 1] / sy if sy != 0 else 0
        shear_y = S[1, 0] / sx if sx != 0 else 0
        return np.array([shear_x, shear_y])

    @property
    def has_perspective(self) -> bool:
        """
        Return True if the homography has a significant perspective component.
        
        Checks if the perspective components p1 = H[2,0] and p2 = H[2,1] 
        are non-zero, indicating a perspective transformation rather than 
        just an affine transformation.
        """
        return not np.allclose(self.value[2, :2], 0)

    @property
    def transform_type(self) -> PerspectiveTransformationMethod:
        return PerspectiveTransformationMethod.HOMOGRAPHY

    @classmethod
    def from_unnormalized_value(
        cls, 
        value: np.ndarray
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
        return cls(value=value / value[2, 2])

    @classmethod
    def create_from_points(
        cls,
        origin_points: np.ndarray, 
        destination_points: np.ndarray, 
        ransac_th: float = 3.0
        ) -> HomographyMatrix:
        """
        Create a homography matrix from a set of origin and destination points.

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
        HomographyMatrix
            Homography matrix.
        """
        if not cls._validate_points(origin_points, destination_points):
            warnings.warn("Invalid points for homography matrix creation")
            return cls.create_identity_matrix()
        
        origin_points = np.asarray(origin_points, dtype=np.float32)
        destination_points = np.asarray(destination_points, dtype=np.float32)
        
        matrix, _ = cv2.findHomography(
            srcPoints=origin_points, 
            dstPoints=destination_points, 
            method=cv2.RANSAC, 
            ransacReprojThreshold=ransac_th
            ) #matrix: shape(3, 3) np.ndarray , mask: shape(n, 1) np.ndarray(bool)

        return cls(value=matrix)

