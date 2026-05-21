# perspective.affine.protocols

## Overview

Affine-only structural types used by `AffineMatrixFactoryMixin` for static typing without importing concrete matrix classes at runtime.

## Components

| Component | Description |
|-----------|-------------|
| [matrix_value.py](./matrix_value.py) | `AffineMatrixValueProtocol` — 2×3 matrix value |
| [matrix_class.py](./matrix_class.py) | `AffineMatrixClassProtocol` — affine factory callable |
| [type_vars.py](./type_vars.py) | `AffineSelfT` type variable |
