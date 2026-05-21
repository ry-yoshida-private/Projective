import cv2
from enum import Enum
from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .matrix import PerspectiveMatrix

class PerspectiveTransformationMethod(Enum):
    """
    Type of perspective transformation matrix.

    Attributes
    ----------
    AFFINE: Uses the affine transformation matrix.
    HOMOGRAPHY: Uses the homography transformation matrix.
    """
    AFFINE = "Affine"
    HOMOGRAPHY = "Homography"

    @property
    def to_cv2_motion_type(self) -> int:
        match self:
            case self.AFFINE:
                return cv2.MOTION_AFFINE
            case self.HOMOGRAPHY:
                return cv2.MOTION_HOMOGRAPHY

    @property
    def perspective_class(self) -> Type['PerspectiveMatrix']:
        """
        Return the perspective class for the transformation method.

        Returns
        -------
        Type['PerspectiveMatrix']:
            The perspective class for the transformation method.
        """
        match self:
            case self.AFFINE:
                from .affine.matrix import AffineMatrix
                return AffineMatrix
            case self.HOMOGRAPHY:
                from .homography.matrix import HomographyMatrix
                return HomographyMatrix
