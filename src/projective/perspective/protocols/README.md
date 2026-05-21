# perspective.protocols

## Overview

Structural typing (`Protocol`) contracts shared by perspective matrix containers and mixins: matrix value shape, class factory surface, 2D point transforms, and decomposition accessors.

## Components

| Component | Description |
|-----------|-------------|
| [matrix_value.py](./matrix_value.py) | `MatrixValueProtocol` — ndarray-backed matrix with `value` |
| [matrix_class.py](./matrix_class.py) | `PerspectiveMatrixClassProtocol` — validation and callable constructor |
| [transform.py](./transform.py) | `PerspectiveMatrixTransformProtocol` — `projective_transformation` |
| [decomposition.py](./decomposition.py) | `PerspectiveMatrixDecompositionProtocol` — translation, rotation, scale, shear |
| [type_vars.py](./type_vars.py) | `PerspectiveSelfT` type variable for mixin methods |
