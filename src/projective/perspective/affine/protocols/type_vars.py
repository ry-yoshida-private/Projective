from __future__ import annotations

from typing import TypeVar

from .matrix_value import AffineMatrixValueProtocol

AffineSelfT = TypeVar("AffineSelfT", bound=AffineMatrixValueProtocol)
