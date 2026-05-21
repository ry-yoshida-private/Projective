from __future__ import annotations

from typing import Protocol

import numpy as np


class MatrixValueProtocol(Protocol):
    """Protocol for matrix containers that expose a ``value`` ndarray."""

    value: np.ndarray
