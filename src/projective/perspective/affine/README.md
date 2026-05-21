# perspective.affine

## Overview

Partial affine (2×3) perspective transforms: typed `AffineMatrix` container, affine-specific protocols, and mixins that call `cv2.estimateAffinePartial2D`.

## Components

| Component | Description |
|-----------|-------------|
| [matrix.py](./matrix.py) | `AffineMatrix` dataclass (shape validation, decomposition overrides) |
| [protocols/](./protocols/README.md) | Affine-specific `Protocol` types for factory typing |
| [mixin/](./mixin/README.md) | Affine estimation and 2×3 point projection mixins |
