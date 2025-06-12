"""
TrustDiff - Enhanced AI Platform Cognitive Assessment with H-CAF Framework.
"""

__version__ = "1.0.0"
__author__ = "TrustDiff Team"
__description__ = "H-CAF Framework for AI Platform Cognitive Assessment"

# Import core models with updated names
from .models import (
    ProbeDefinition, RawResult, EvaluationResult, PlatformConfig, 
    QualityEvaluation, DetailedScores, CognitiveFingerprint, CapabilityGaps,
    RunConfig, TrustDiffReport, ExecutionPlan
)

# Import core components  
from .engine import Engine
from .comparator import HCAFComparator, Comparator  # Keep backward compatibility
from .storage import TrustDiffStorage

# Create aliases for consistent naming
TrustDiffEngine = Engine  # Alias for the main engine class
Storage = TrustDiffStorage  # Backward compatibility
Probe = ProbeDefinition  # Backward compatibility

__all__ = [
    # Core models
    "ProbeDefinition",
    "RawResult", 
    "EvaluationResult",
    "PlatformConfig",
    "QualityEvaluation",
    "DetailedScores",
    "CognitiveFingerprint",
    "CapabilityGaps",
    "RunConfig",
    "TrustDiffReport",
    "ExecutionPlan",
    
    # Core components
    "Engine",
    "TrustDiffEngine",
    "HCAFComparator",
    "TrustDiffStorage",
    
    # Backward compatibility
    "Comparator", 
    "Storage",
    "Probe",
] 