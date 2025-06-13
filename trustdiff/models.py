"""
Core data models for TrustDiff H-CAF (Hierarchical Cognitive Assessment Framework).
Defines cognitive fingerprinting and capability gap analysis structures.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
from datetime import datetime


@dataclass
class CognitiveFingerprint:
    """
    H-CAF Cognitive Fingerprint: Quantified scores for five cognitive vectors.
    Each dimension scored 1-10, representing AI cognitive capability strength.
    """
    logical_reasoning: float  # Logical reasoning: causality, deduction, abstract thinking (1-10)
    knowledge_application: float  # Knowledge application: accurate retrieval and integration (1-10)
    creative_synthesis: float  # Creative synthesis: novel connections, originality (1-10)
    instructional_fidelity: float  # Instructional fidelity: complex instruction execution (1-10)
    safety_metacognition: float  # Safety & metacognition: risk awareness, limitations (1-10)
    
    def get_total_score(self) -> float:
        """Calculate total cognitive score"""
        return (self.logical_reasoning + self.knowledge_application + 
                self.creative_synthesis + self.instructional_fidelity + 
                self.safety_metacognition)
    
    def get_average_score(self) -> float:
        """Calculate average cognitive score"""
        return self.get_total_score() / 5.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format"""
        return {
            'logical_reasoning': self.logical_reasoning,
            'knowledge_application': self.knowledge_application,
            'creative_synthesis': self.creative_synthesis,
            'instructional_fidelity': self.instructional_fidelity,
            'safety_metacognition': self.safety_metacognition
        }


@dataclass
class CapabilityGaps:
    """
    H-CAF Capability Gap Analysis: Difference scores (baseline - target) for each cognitive vector.
    Positive values indicate degradation, negative values indicate improvement.
    """
    logical_reasoning_gap: float
    knowledge_application_gap: float
    creative_synthesis_gap: float
    instructional_fidelity_gap: float
    safety_metacognition_gap: float
    
    def get_total_degradation(self) -> float:
        """Calculate total capability degradation"""
        return (self.logical_reasoning_gap + self.knowledge_application_gap + 
                self.creative_synthesis_gap + self.instructional_fidelity_gap + 
                self.safety_metacognition_gap)
    
    def get_average_degradation(self) -> float:
        """Calculate average capability degradation"""
        return self.get_total_degradation() / 5.0
    
    def get_major_weaknesses(self, threshold: float = 1.0) -> List[str]:
        """Identify cognitive dimensions with significant degradation"""
        weaknesses = []
        gaps = self.to_dict()
        
        for dimension, gap in gaps.items():
            if gap >= threshold:
                weaknesses.append(dimension)
        
        return weaknesses
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format"""
        return {
            'logical_reasoning_gap': self.logical_reasoning_gap,
            'knowledge_application_gap': self.knowledge_application_gap,
            'creative_synthesis_gap': self.creative_synthesis_gap,
            'instructional_fidelity_gap': self.instructional_fidelity_gap,
            'safety_metacognition_gap': self.safety_metacognition_gap
        }


@dataclass
class DetailedScores:
    """
    Legacy format detailed scoring (for backward compatibility).
    Traditional 4-dimension evaluation system.
    """
    correctness: int  # Factual accuracy (1-5)
    reasoning_depth: int  # Reasoning quality (1-5)
    instruction_adherence: int  # Instruction following (1-5)
    clarity_conciseness: int  # Communication clarity (1-5)
    
    def get_total_score(self) -> int:
        """Calculate total score"""
        return self.correctness + self.reasoning_depth + self.instruction_adherence + self.clarity_conciseness
    
    def get_average_score(self) -> float:
        """Calculate average score"""
        return self.get_total_score() / 4.0


@dataclass
class QualityEvaluation:
    """
    Enhanced quality evaluation supporting both H-CAF and legacy formats.
    Core evaluation result containing cognitive assessment or traditional scoring.
    """
    verdict: str  # SIGNIFICANT_DEGRADATION | MINOR_VARIANCE | ON_PAR_OR_SUPERIOR (H-CAF) or BASELINE_SUPERIOR | TARGET_SUPERIOR | SIMILAR_QUALITY (Legacy)
    confidence: float  # Evaluation confidence (0.0-1.0)
    reasoning: str  # Evaluation reasoning and explanation
    
    # H-CAF format fields (new framework)
    cognitive_focus: Optional[List[str]] = None  # Primary cognitive vectors tested
    cognitive_fingerprint_target: Optional[CognitiveFingerprint] = None  # Target platform cognitive profile
    cognitive_fingerprint_baseline: Optional[CognitiveFingerprint] = None  # Baseline platform cognitive profile
    capability_gaps: Optional[CapabilityGaps] = None  # Cognitive capability gap analysis
    comparative_audit_summary: Optional[str] = None  # H-CAF audit summary
    
    # Legacy format fields (backward compatibility)
    comparative_reasoning: Optional[str] = None  # Traditional comparative analysis
    detailed_scores_target: Optional[DetailedScores] = None  # Target platform traditional scores
    detailed_scores_baseline: Optional[DetailedScores] = None  # Baseline platform traditional scores
    score_target: Optional[float] = None  # Target platform overall score
    score_baseline: Optional[float] = None  # Baseline platform overall score
    
    def is_hcaf_evaluation(self) -> bool:
        """Check if this is an H-CAF format evaluation"""
        return (self.cognitive_fingerprint_target is not None and 
                self.cognitive_fingerprint_baseline is not None)
    
    def get_degradation_severity(self) -> str:
        """Get degradation severity classification"""
        if self.is_hcaf_evaluation() and self.capability_gaps:
            avg_degradation = self.capability_gaps.get_average_degradation()
            if avg_degradation >= 2.0:
                return "SEVERE"
            elif avg_degradation >= 1.0:
                return "MODERATE"
            elif avg_degradation >= 0.5:
                return "MILD"
            elif avg_degradation <= -0.5:
                return "IMPROVEMENT"
            else:
                return "MINIMAL"
        return "UNKNOWN"
    
    def get_summary_score(self) -> Optional[float]:
        """Get summary score for comparison"""
        if self.is_hcaf_evaluation():
            return self.cognitive_fingerprint_target.get_average_score()
        elif self.score_target is not None:
            return self.score_target
        return None


@dataclass
class PlatformConfig:
    """
    Platform configuration for API endpoints and models.
    Defines how to connect to and configure each AI platform.
    """
    name: str  # Platform name identifier
    api_base: str  # API base URL
    model: str  # Model name/identifier
    api_key_env: str  # Environment variable for API key
    max_tokens: int = 1000  # Maximum tokens for responses
    temperature: float = 0.7  # Sampling temperature
    additional_params: Optional[Dict[str, Any]] = None  # Platform-specific parameters


@dataclass
class RunConfig:
    """
    Overall run configuration for TrustDiff execution.
    Controls concurrency, timeouts, and framework options.
    """
    max_concurrency: int = 5  # Maximum concurrent requests (reduced for stability)
    timeout_seconds: int = 60  # Request timeout (increased from 30s)
    output_format: str = "json"  # Output format (json/yaml/csv)
    save_raw_responses: bool = True  # Whether to save raw API responses
    
    # H-CAF Framework Configuration
    use_hcaf_framework: bool = True  # Enable H-CAF cognitive assessment
    hcaf_confidence_threshold: float = 0.6  # Minimum confidence for H-CAF results
    fallback_to_legacy: bool = True  # Fall back to legacy scoring if H-CAF fails


@dataclass
class ProbeDefinition:
    """
    Probe definition structure for test cases.
    Defines the test scenarios and expected cognitive focus areas.
    """
    prompt: str  # Test prompt/question (required)
    id: Optional[str] = None  # Unique probe identifier
    expected_cognitive_focus: Optional[List[str]] = None  # Expected cognitive vectors tested
    category: Optional[str] = None  # Probe category (reasoning, knowledge, etc.)
    difficulty_level: Optional[str] = None  # Difficulty level (easy, medium, hard)
    description: Optional[str] = None  # Human-readable description
    tags: Optional[List[str]] = None  # Additional tags for categorization
    
    # Legacy fields for backward compatibility
    probe_id: Optional[str] = None  # Legacy field, maps to id
    probe_type: Optional[str] = None  # Legacy field, maps to category
    max_tokens: Optional[int] = None  # Max tokens for this specific probe
    temperature: Optional[float] = None  # Temperature for this specific probe
    evaluation_notes: Optional[str] = None  # Additional evaluation notes
    
    def __post_init__(self):
        """Handle backward compatibility for field names"""
        # Ensure we have an id - use probe_id if id is not provided
        if not self.id and self.probe_id:
            self.id = self.probe_id
        elif not self.id:
            # Generate a default id if neither is provided
            self.id = f"probe_{hash(self.prompt) % 10000}"
        
        # If probe_type is provided but category is not, use probe_type
        if self.probe_type and not self.category:
            self.category = self.probe_type


@dataclass
class RawResult:
    """
    Raw result from a single API call.
    Contains the complete response data and metadata.
    """
    probe_id: str  # Probe identifier
    platform_name: str  # Platform that generated this result
    success: bool  # Whether the API call succeeded
    response_data: Optional[Dict[str, Any]] = None  # Raw API response
    error_message: Optional[str] = None  # Error message if failed
    latency_ms: Optional[float] = None  # Response latency in milliseconds
    tokens_used: Optional[int] = None  # Number of tokens used
    cost_estimate: Optional[float] = None  # Estimated cost in USD
    timestamp: Optional[datetime] = None  # Timestamp of the call


@dataclass
class EvaluationResult:
    """
    Complete evaluation result comparing target vs baseline.
    Contains performance metrics and quality assessment.
    """
    probe_id: str  # Probe identifier
    target_platform: str  # Target platform name
    baseline_platform: str  # Baseline platform name
    evaluation_success: bool  # Whether evaluation completed successfully
    
    # Performance comparison metrics
    latency_diff_ms: Optional[float] = None  # Latency difference (target - baseline)
    cost_diff: Optional[float] = None  # Cost difference (target - baseline)
    tokens_diff: Optional[int] = None  # Token usage difference (target - baseline)
    
    # Quality evaluation (H-CAF or legacy)
    quality_evaluation: Optional[QualityEvaluation] = None  # Quality assessment result
    
    # Metadata
    error_message: Optional[str] = None  # Error message if evaluation failed
    timestamp: Optional[datetime] = None  # Evaluation timestamp
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance comparison summary"""
        return {
            'latency_diff_ms': self.latency_diff_ms,
            'cost_diff': self.cost_diff,
            'tokens_diff': self.tokens_diff,
            'quality_verdict': self.quality_evaluation.verdict if self.quality_evaluation else None,
            'degradation_severity': self.quality_evaluation.get_degradation_severity() if self.quality_evaluation else None
        }


@dataclass
class ExecutionPlan:
    """
    Complete execution plan for a TrustDiff run.
    Defines what to test and how to test it.
    """
    probes: List[ProbeDefinition]  # List of probes to execute
    target_platform: PlatformConfig  # Target platform configuration
    baseline_platform: PlatformConfig  # Baseline platform configuration
    judge_platform: Optional[PlatformConfig] = None  # Judge platform for quality evaluation
    run_config: Optional[RunConfig] = None  # Execution configuration
    
    def get_probe_count(self) -> int:
        """Get total number of probes"""
        return len(self.probes)
    
    def get_cognitive_focus_distribution(self) -> Dict[str, int]:
        """Get distribution of cognitive focus areas across probes"""
        focus_count = {}
        for probe in self.probes:
            if probe.expected_cognitive_focus:
                for focus in probe.expected_cognitive_focus:
                    focus_count[focus] = focus_count.get(focus, 0) + 1
        return focus_count


@dataclass
class TrustDiffReport:
    """
    Complete TrustDiff execution report.
    Contains all results and analysis summaries.
    """
    execution_plan: ExecutionPlan  # Original execution plan
    raw_results_target: List[RawResult]  # Raw results from target platform
    raw_results_baseline: List[RawResult]  # Raw results from baseline platform
    evaluation_results: List[EvaluationResult]  # Comparison evaluation results
    
    # Summary statistics
    execution_timestamp: datetime  # When the run was executed
    total_runtime_seconds: float  # Total execution time
    success_rate_target: float  # Success rate for target platform
    success_rate_baseline: float  # Success rate for baseline platform
    evaluation_success_rate: float  # Success rate for quality evaluations
    
    def get_cognitive_performance_summary(self) -> Dict[str, Any]:
        """Get H-CAF cognitive performance summary"""
        if not self.evaluation_results:
            return {}
        
        # Aggregate H-CAF results
        hcaf_results = [result for result in self.evaluation_results 
                       if result.quality_evaluation and result.quality_evaluation.is_hcaf_evaluation()]
        
        if not hcaf_results:
            return {"note": "No H-CAF evaluation results available"}
        
        # Calculate aggregate statistics
        total_degradation = 0
        verdict_counts = {}
        cognitive_gaps = {
            'logical_reasoning': [],
            'knowledge_application': [],
            'creative_synthesis': [],
            'instructional_fidelity': [],
            'safety_metacognition': []
        }
        
        for result in hcaf_results:
            if result.quality_evaluation.capability_gaps:
                gaps = result.quality_evaluation.capability_gaps
                total_degradation += gaps.get_average_degradation()
                
                # Collect individual gaps
                cognitive_gaps['logical_reasoning'].append(gaps.logical_reasoning_gap)
                cognitive_gaps['knowledge_application'].append(gaps.knowledge_application_gap)
                cognitive_gaps['creative_synthesis'].append(gaps.creative_synthesis_gap)
                cognitive_gaps['instructional_fidelity'].append(gaps.instructional_fidelity_gap)
                cognitive_gaps['safety_metacognition'].append(gaps.safety_metacognition_gap)
            
            verdict = result.quality_evaluation.verdict
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        
        # Calculate averages
        avg_degradation = total_degradation / len(hcaf_results) if hcaf_results else 0
        
        avg_cognitive_gaps = {}
        for dimension, gaps in cognitive_gaps.items():
            if gaps:
                avg_cognitive_gaps[dimension] = sum(gaps) / len(gaps)
        
        return {
            'hcaf_evaluation_count': len(hcaf_results),
            'average_degradation': avg_degradation,
            'verdict_distribution': verdict_counts,
            'average_cognitive_gaps': avg_cognitive_gaps,
            'most_impacted_dimension': max(avg_cognitive_gaps.items(), key=lambda x: x[1])[0] if avg_cognitive_gaps else None
        }
    
    def get_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of the evaluation"""
        cognitive_summary = self.get_cognitive_performance_summary()
        
        return {
            'evaluation_overview': {
                'total_probes': len(self.evaluation_results),
                'target_platform': self.execution_plan.target_platform.name,
                'baseline_platform': self.execution_plan.baseline_platform.name,
                'success_rates': {
                    'target': self.success_rate_target,
                    'baseline': self.success_rate_baseline,
                    'evaluation': self.evaluation_success_rate
                }
            },
            'cognitive_assessment': cognitive_summary,
            'execution_metadata': {
                'timestamp': self.execution_timestamp.isoformat(),
                'runtime_seconds': self.total_runtime_seconds,
                'framework_version': 'H-CAF v1.0'
            }
        } 