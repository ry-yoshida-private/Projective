import cv2
from enum import Enum

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
