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
    # H-CAF: 认知向量标签
    cognitive_focus: Optional[List[str]] = None  # 主要测试的认知向量
    
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


# H-CAF: 认知指纹模型
class CognitiveFingerprint(BaseModel):
    """H-CAF认知指纹：5个核心认知向量的评分"""
    logical_reasoning: int = Field(ge=1, le=10, description="逻辑推理能力 (1-10)")
    knowledge_application: int = Field(ge=1, le=10, description="知识应用能力 (1-10)")
    creative_synthesis: int = Field(ge=1, le=10, description="创造性综合能力 (1-10)")
    instructional_fidelity: int = Field(ge=1, le=10, description="指令忠实度 (1-10)")
    safety_metacognition: int = Field(ge=1, le=10, description="安全与元认知能力 (1-10)")
    
    def get_total_score(self) -> int:
        """计算总认知得分"""
        return (self.logical_reasoning + self.knowledge_application + 
                self.creative_synthesis + self.instructional_fidelity + 
                self.safety_metacognition)
    
    def get_average_score(self) -> float:
        """计算平均认知得分"""
        return self.get_total_score() / 5.0


class CapabilityGaps(BaseModel):
    """H-CAF能力差距分析"""
    logical_reasoning_gap: float = Field(description="逻辑推理能力差距 (baseline - target)")
    knowledge_application_gap: float = Field(description="知识应用能力差距")
    creative_synthesis_gap: float = Field(description="创造性综合能力差距")
    instructional_fidelity_gap: float = Field(description="指令忠实度差距")
    safety_metacognition_gap: float = Field(description="安全与元认知能力差距")
    
    def get_overall_degradation(self) -> float:
        """计算整体能力退化程度"""
        gaps = [
            self.logical_reasoning_gap, self.knowledge_application_gap,
            self.creative_synthesis_gap, self.instructional_fidelity_gap,
            self.safety_metacognition_gap
        ]
        return sum(gap for gap in gaps if gap > 0) / 5.0


# 保留旧的DetailedScores以向后兼容
class DetailedScores(BaseModel):
    """Detailed evaluation scores for a single answer (Legacy compatibility)."""
    correctness: int = Field(ge=1, le=5)
    reasoning_depth: int = Field(ge=1, le=5)
    instruction_adherence: int = Field(ge=1, le=5)
    clarity_conciseness: int = Field(ge=1, le=5)


class QualityEvaluation(BaseModel):
    """H-CAF质量评估结果"""
    # H-CAF新字段
    cognitive_focus: Optional[List[str]] = None  # 探针主要测试的认知向量
    cognitive_fingerprint_target: Optional[CognitiveFingerprint] = None
    cognitive_fingerprint_baseline: Optional[CognitiveFingerprint] = None
    capability_gaps: Optional[CapabilityGaps] = None
    comparative_audit_summary: Optional[str] = None
    
    # 新的判决标准
    verdict: Literal[
        "SIGNIFICANT_DEGRADATION",  # 显著退化
        "MINOR_VARIANCE",          # 轻微差异
        "ON_PAR_OR_SUPERIOR",      # 同等或更优
        # 保留旧格式以向后兼容
        "BASELINE_SUPERIOR", "TARGET_SUPERIOR", "SIMILAR_QUALITY", 
        "baseline_better", "target_better", "equivalent", "both_poor"
    ]
    
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str
    
    # Legacy字段（向后兼容）
    score_baseline: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    score_target: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    detailed_scores_target: Optional[DetailedScores] = None
    detailed_scores_baseline: Optional[DetailedScores] = None
    comparative_reasoning: Optional[str] = None
    
    def get_degradation_severity(self) -> str:
        """获取退化严重程度的描述"""
        if not self.capability_gaps:
            return "Unknown"
        
        overall_degradation = self.capability_gaps.get_overall_degradation()
        if overall_degradation >= 3.0:
            return "Severe"
        elif overall_degradation >= 1.5:
            return "Moderate"
        elif overall_degradation >= 0.5:
            return "Mild"
        else:
            return "Minimal"


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
    
    # H-CAF质量评估
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
    timeout_seconds: int = 60  # 默认60秒超时
    
    # H-CAF设置
    use_hcaf_framework: bool = Field(default=True, description="是否使用H-CAF认知评估框架")
    cognitive_vectors_focus: Optional[List[str]] = Field(
        default=None, 
        description="重点关注的认知向量，None表示使用全部5个向量"
    )
    
    # API keys
    api_keys: Dict[str, str] = Field(default_factory=dict)


class CognitiveBenchmarkSummary(BaseModel):
    """H-CAF认知基准测试摘要"""
    total_cognitive_evaluations: int
    platform_cognitive_profiles: Dict[str, Dict[str, float]]  # 平台 -> 认知向量 -> 平均得分
    most_degraded_vectors: List[str]  # 退化最严重的认知向量
    overall_degradation_score: float  # 整体退化评分
    cognitive_stability_rating: str  # 认知稳定性评级


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
    
    # H-CAF摘要
    cognitive_benchmark_summary: Optional[CognitiveBenchmarkSummary] = None 