"""Static specialist composition with explicit, empirical capability contracts."""
from .artifacts import Artifact, CompatibilityError, export_artifact, validate_compatibility
from .models import MLP
from .merge import merge
from .repair import BuildBudget, Capability, repair

__all__ = ["Artifact", "CompatibilityError", "export_artifact", "validate_compatibility", "MLP", "merge", "BuildBudget", "Capability", "repair"]
__version__ = "0.2.0"
