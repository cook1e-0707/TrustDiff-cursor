"""
Data models for TrustDiff using Pydantic.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, validator


class ProbeMessage(BaseModel):
    """Individual message in a probe conversation."""
    role: Literal["system", "user", "assistant"]
    content: str


class Probe(BaseModel):
    """Test probe definition."""
    probe_id: str
    probe_type: Literal["reasoning", "cost", "quality", "safety"]
    description: str
    prompt: List[ProbeMessage]
    expected_model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    
    @validator('probe_id')
    def validate_probe_id(cls, v):
        if not v or not v.strip():
            raise ValueError("probe_id cannot be empty")
        return v.strip()


class PlatformConfig(BaseModel):
    """Configuration for an LLM platform."""
    name: str
    api_base: str
    api_key_env: str
    model: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    # Advanced options for third-party platforms
    auth_type: Optional[str] = "bearer"  # bearer, api_key, custom
    api_key_header: Optional[str] = "Authorization"  # Custom header name for API key
    api_key_prefix: Optional[str] = "Bearer"  # Prefix for API key (Bearer, sk-, etc.)
    endpoint_path: Optional[str] = "/chat/completions"  # Custom endpoint path
    request_format: Optional[str] = "openai"  # openai, custom


class RawResult(BaseModel):
    """Raw API response result."""
    probe_id: str
    platform_name: str
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool
    response_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    latency_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None


class DetailedScores(BaseModel):
    """Detailed evaluation scores for a single answer."""
    correctness: int = Field(ge=1, le=5)
    reasoning_depth: int = Field(ge=1, le=5)
    instruction_adherence: int = Field(ge=1, le=5)
    clarity_conciseness: int = Field(ge=1, le=5)


class QualityEvaluation(BaseModel):
    """Quality evaluation result from LLM judge."""
    verdict: Literal["BASELINE_SUPERIOR", "TARGET_SUPERIOR", "SIMILAR_QUALITY", "baseline_better", "target_better", "equivalent", "both_poor"]
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str
    
    # Legacy simple scores (for backward compatibility)
    score_baseline: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    score_target: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    
    # New detailed scores
    detailed_scores_target: Optional[DetailedScores] = None
    detailed_scores_baseline: Optional[DetailedScores] = None
    comparative_reasoning: Optional[str] = None


class EvaluationResult(BaseModel):
    """Complete evaluation result comparing baseline and target."""
    probe_id: str
    target_platform: str
    baseline_platform: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Performance metrics
    latency_diff_ms: Optional[float] = None
    cost_diff: Optional[float] = None
    tokens_diff: Optional[int] = None
    
    # Quality evaluation
    quality_evaluation: Optional[QualityEvaluation] = None
    
    # Status
    evaluation_success: bool = True
    error_message: Optional[str] = None


class RunConfig(BaseModel):
    """Configuration for a test run."""
    baseline: PlatformConfig
    targets: List[PlatformConfig]
    judge: Optional[PlatformConfig] = None
    
    # Run settings
    probe_dir: str = "./probes"
    output_dir: str = "./outputs"
    concurrency: int = 10
    timeout_seconds: int = 60
    
    # API keys
    api_keys: Dict[str, str] = Field(default_factory=dict)


class TestSummary(BaseModel):
    """Summary of a complete test run."""
    run_id: str
    timestamp: datetime
    total_probes: int
    total_platforms: int
    total_evaluations: int
    success_rate: float
    average_latency_diff: Optional[float] = None
    average_cost_diff: Optional[float] = None
    quality_summary: Dict[str, int] = Field(default_factory=dict)
    output_dir: Optional[str] = None 