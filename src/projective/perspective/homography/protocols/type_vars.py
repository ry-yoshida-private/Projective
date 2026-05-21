from __future__ import annotations

from typing import TypeVar

from .matrix_value import HomographyMatrixValueProtocol

HomographySelfT = TypeVar("HomographySelfT", bound=HomographyMatrixValueProtocol)
