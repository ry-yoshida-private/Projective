from __future__ import annotations

from typing import TypeVar

from .matrix_value import MatrixValueProtocol

PerspectiveSelfT = TypeVar("PerspectiveSelfT", bound=MatrixValueProtocol)
