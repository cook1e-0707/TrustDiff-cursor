"""
Simple tests for TrustDiff models.
"""

import pytest
from datetime import datetime
from trustdiff.models import Probe, ProbeMessage, PlatformConfig, RawResult, QualityEvaluation


def test_probe_creation():
    """Test creating a probe."""
    messages = [ProbeMessage(role="user", content="Test message")]
    probe = Probe(
        probe_id="test_probe",
        probe_type="reasoning",
        description="A test probe",
        prompt=messages
    )
    
    assert probe.probe_id == "test_probe"
    assert probe.probe_type == "reasoning"
    assert len(probe.prompt) == 1
    assert probe.prompt[0].content == "Test message"


def test_platform_config():
    """Test platform configuration."""
    config = PlatformConfig(
        name="Test Platform",
        api_base="https://api.test.com/v1",
        api_key_env="TEST_KEY",
        model="gpt-3.5-turbo"
    )
    
    assert config.name == "Test Platform"
    assert config.api_base == "https://api.test.com/v1"
    assert config.model == "gpt-3.5-turbo"


def test_raw_result():
    """Test raw result creation."""
    result = RawResult(
        probe_id="test_probe",
        platform_name="Test Platform",
        success=True,
        response_data={"test": "data"},
        latency_ms=150.5
    )
    
    assert result.probe_id == "test_probe"
    assert result.success is True
    assert result.latency_ms == 150.5
    assert isinstance(result.timestamp, datetime)


def test_quality_evaluation():
    """Test quality evaluation."""
    evaluation = QualityEvaluation(
        verdict="baseline_better",
        confidence=0.85,
        reasoning="The baseline response was more accurate"
    )
    
    assert evaluation.verdict == "baseline_better"
    assert evaluation.confidence == 0.85
    assert "accurate" in evaluation.reasoning


if __name__ == "__main__":
    pytest.main([__file__]) 