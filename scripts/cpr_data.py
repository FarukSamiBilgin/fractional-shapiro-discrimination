"""Current-phase-relation data shared by the thermal, node and R23 scripts.

The harmonics live in ``data/beenakker_cpr.json`` (short-junction Beenakker CPR
at transparency 0.8, already normalised to max I = 1). In the original working
tree they were read from a sibling project's material file; they are vendored
here so this repository is self-contained.
"""

from __future__ import annotations

import json
import os

import numpy as np

from shapiro import normalize_cpr

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data", "beenakker_cpr.json")

# Reference drive frequency (units of the characteristic frequency) used by
# r23_alpha.py.
OMEGA_0 = 5.0


def load_cpr(tau: str = "0.8", path: str = _DATA) -> np.ndarray:
    with open(path) as fh:
        mat = json.load(fh)
    return normalize_cpr(mat["harmonics_by_tau"][tau])
