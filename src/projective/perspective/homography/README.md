# perspective.homography

## Overview

Planar homography (3×3) perspective transforms: typed `HomographyMatrix` container, homography-specific protocols, and mixins that call `cv2.findHomography`.

## Components

| Component | Description |
|-----------|-------------|
| [matrix.py](./matrix.py) | `HomographyMatrix` dataclass (shape validation, perspective components) |
| [protocols/](./protocols/README.md) | Homography-specific `Protocol` types for factory typing |
| [mixin/](./mixin/README.md) | Homography estimation and dehomogenized point projection mixins |
