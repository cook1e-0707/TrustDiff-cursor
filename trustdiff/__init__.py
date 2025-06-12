"""
TrustDiff - A Python testing framework for detecting semantic vulnerabilities in LLM API platforms.
"""

__version__ = "0.1.0"
__author__ = "TrustDiff Team"
__description__ = "Testing framework for LLM API platform reliability"

from .models import Probe, RawResult, EvaluationResult, PlatformConfig, QualityEvaluation, DetailedScores
from .engine import Engine
from .comparator import Comparator
from .storage import Storage
from .reporter import Reporter

__all__ = [
    "Probe",
    "RawResult", 
    "EvaluationResult",
    "PlatformConfig",
    "QualityEvaluation",
    "DetailedScores",
    "Engine",
    "Comparator",
    "Storage",
    "Reporter",
] 