"""
Off-axis / anisotropic multilayer simulation tools (4x4 TMM).

Public API
----------
- Geometry
- UniaxialMaterial
- TMM
- LayerStack + builders
- SingleLayerSimulation, MultiLayerSimulation
- Jones <-> Mueller utilities
"""

# -------------------------
# Package metadata
# -------------------------
__all__ = [
    "Geometry",
    "UniaxialMaterial",
    "TMM",
    "LayerStack",
    "build_single_layer",
    "build_hwp_stack",
    "build_klopfenstein_arc",
    "wrap_with_symmetric_arc",
    "SingleLayerSimulation",
    "MultiLayerSimulation",
    "jones_to_mueller",
    "jones_to_mueller_from_elements",
]

__version__ = "0.1.0"
__author__ = "Yurika Hoshino; Ryota Takaku"
__license__ = ""
__description__ = "4x4 TMM simulator for uniaxial multilayer stacks (Jones/Mueller)."
__url__ = ""


# -------------------------
# Public imports
# -------------------------
from .geometry import Geometry
from .material import UniaxialMaterial
from .tmm import TMM

from .layers import (
    LayerStack,
    build_single_layer,
    build_hwp_stack,
    build_klopfenstein_arc,
    wrap_with_symmetric_arc,
)

from .simulation import (
    SingleLayerSimulation,
    MultiLayerSimulation,
)

from .jones_mueller import (
    jones_to_mueller,
    jones_to_mueller_from_elements,
)
