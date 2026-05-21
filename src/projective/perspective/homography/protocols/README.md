# perspective.homography.protocols

## Overview

Homography-only structural types used by `HomographyFactoryMixin` for static typing without circular imports.

## Components

| Component | Description |
|-----------|-------------|
| [matrix_value.py](./matrix_value.py) | `HomographyMatrixValueProtocol` — 3×3 matrix value |
| [matrix_class.py](./matrix_class.py) | `HomographyMatrixClassProtocol` — homography factory callable |
| [inverse.py](./inverse.py) | `HomographyInverseProtocol` — `inverse` for projective mixin typing |
| [type_vars.py](./type_vars.py) | `HomographySelfT` type variable |
