"""MonoGeoFlightLog: monocular geometry for aerial and driving flight logs.

Three two-view geometry families are implemented and compared on identical
data:

1. **Classical projective geometry** -- checkerboard calibration, SIFT
   matching, essential matrix, pose recovery and triangulation
   (:mod:`monogeo.calib`, :mod:`monogeo.twoview`).
2. **Self-supervised monocular depth + ego-motion** -- a Monodepth2-style CNN
   trained only with a photometric loss through a differentiable camera model
   (:mod:`monogeo.models.depth_net`, :mod:`monogeo.models.pose_net`,
   :mod:`monogeo.geometry`, :mod:`monogeo.losses`).
3. **End-to-end two-view pointmap regression** -- a compact DUSt3R/MASt3R-style
   network that emits per-pixel 3D pointmaps and a confidence map
   (:mod:`monogeo.models.pointmap_net`, :mod:`monogeo.pointmap_ops`).

Everything runs offline on CPU thanks to the deterministic synthetic renderer
in :mod:`monogeo.synth`.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
