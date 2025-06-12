#!/usr/bin/env python3
"""
H-CAF JSON Parsing Test Script
Tests the enhanced JSON parsing fixes for the H-CAF framework.
"""

import asyncio
import json
import re
from trustdiff.comparator import HCAFComparator
from trustdiff.models import PlatformConfig, RawResult


# Simulated Gemini responses with common JSON issues
PROBLEMATIC_RESPONSES = [
    # Issue 1: Trailing commas
    '''
{
  "cognitive_fingerprint": {
    "answer_a": {
      "logical_reasoning": 7,
      "knowledge_application": 8,
      "creative_synthesis": 6,
      "instructional_fidelity": 9,
      "safety_metacognition": 7,
    },
    "answer_b": {
      "logical_reasoning": 6,
      "knowledge_application": 7,
      "creative_synthesis": 5,
      "instructional_fidelity": 8,
      "safety_metacognition": 6,
    }
  },
  "capability_gaps": {
    "logical_reasoning_gap": -1,
    "knowledge_application_gap": -1,
    "creative_synthesis_gap": -1,
    "instructional_fidelity_gap": -1,
    "safety_metacognition_gap": -1,
  },
  "comparative_audit_summary": "Answer A shows better performance across all dimensions",
  "final_verdict": "ON_PAR_OR_SUPERIOR"
}
    ''',
    
    # Issue 2: Comments and placeholders
    '''
{
  "cognitive_fingerprint": {
    "answer_a": {
      "logical_reasoning": <8>,  // Good reasoning ability
      "knowledge_application": <7>,
      "creative_synthesis": <6>,
      "instructional_fidelity": <9>,
      "safety_metacognition": <7>
    },
    "answer_b": {
      "logical_reasoning": <6>,
      "knowledge_application": <7>,
      "creative_synthesis": <5>,
      "instructional_fidelity": <8>,
      "safety_metacognition": <6>
    }
  },
  "capability_gaps": {
    "logical_reasoning_gap": -2,
    "knowledge_application_gap": 0,
    "creative_synthesis_gap": -1,
    "instructional_fidelity_gap": -1,
    "safety_metacognition_gap": -1
  },
  "comparative_audit_summary": "Target shows improvement in reasoning",
  "final_verdict": ON_PAR_OR_SUPERIOR  // Unquoted string
}
    ''',
    
    # Issue 3: Mixed with text explanation
    '''
Based on the cognitive assessment, here is my evaluation:

{
  "cognitive_fingerprint": {
    "answer_a": {
      "logical_reasoning": 6,
      "knowledge_application": 8,
      "creative_synthesis": 7,
      "instructional_fidelity": 9,
      "safety_metacognition": 8
    },
    "answer_b": {
      "logical_reasoning": 7,
      "knowledge_application": 8,
      "creative_synthesis": 6,
      "instructional_fidelity": 8,
      "safety_metacognition": 7
    }
  },
  "capability_gaps": {
    "logical_reasoning_gap": 1,
    "knowledge_application_gap": 0,
    "creative_synthesis_gap": -1,
    "instructional_fidelity_gap": -1,
    "safety_metacognition_gap": -1
  },
  "comparative_audit_summary": "Mixed performance with some improvements",
  "final_verdict": "MINOR_VARIANCE"
}

This assessment shows balanced cognitive performance.
    ''',
    
    # Issue 4: Severely broken JSON
    '''
{
  "cognitive_fingerprint": {
    "answer_a": {
      "logical_reasoning": 5,
      "knowledge_application": 6,
      "creative_synthesis": 4,
      "instructional_fidelity": 7,
      "safety_metacognition": 6,
    },
    "answer_b": {
      "logical_reasoning": 8,
      "knowledge_application": 9,
      "creative_synthesis": 7,
      "instructional_fidelity": 9,
      "safety_metacognition": 8
    }
  },
  "capability_gaps": {
    "logical_reasoning_gap": 3,
    "knowledge_application_gap": 3,
    "creative_synthesis_gap": 3,
    "instructional_fidelity_gap": 2,
    "safety_metacognition_gap": 2,
  },
  "comparative_audit_summary": "Significant degradation detected",
  "final_verdict": SIGNIFICANT_DEGRADATION
}
    '''
]


async def test_json_parsing():
    """Test JSON parsing with various problematic responses"""
    
    print("🧪 Testing H-CAF JSON Parsing Fixes")
    print("=" * 50)
    
    # Create a mock comparator
    judge_config = PlatformConfig(
        name="test_judge",
        api_base="https://api.openai.com/v1",
        model="gpt-4",
        api_key_env="OPENAI_API_KEY"
    )
    
    comparator = HCAFComparator(
        judge_config=judge_config,
        api_keys={"OPENAI_API_KEY": "test"},
        use_hcaf=True
    )
    
    success_count = 0
    total_tests = len(PROBLEMATIC_RESPONSES)
    
    for i, problematic_json in enumerate(PROBLEMATIC_RESPONSES, 1):
        print(f"\n🔍 Test Case {i}: Testing JSON parsing...")
        
        # Create mock response data
        mock_response = {
            'choices': [
                {
                    'message': {
                        'content': problematic_json
                    }
                }
            ]
        }
        
        try:
            # Test the parsing method
            evaluation = comparator._parse_hcaf_response(mock_response)
            
            if evaluation and evaluation.cognitive_fingerprint_target:
                print(f"✅ Test Case {i}: PASSED - H-CAF evaluation parsed successfully")
                print(f"   Verdict: {evaluation.verdict}")
                print(f"   Target cognitive score: {evaluation.cognitive_fingerprint_target.get_average_score():.2f}")
                if evaluation.capability_gaps:
                    print(f"   Average degradation: {evaluation.capability_gaps.get_average_degradation():.2f}")
                success_count += 1
            else:
                print(f"⚠️  Test Case {i}: PARTIAL - Fell back to legacy/fallback parsing")
                if evaluation:
                    print(f"   Verdict: {evaluation.verdict}")
                    print(f"   Confidence: {evaluation.confidence}")
        
        except Exception as e:
            print(f"❌ Test Case {i}: FAILED - Exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 TEST RESULTS SUMMARY")
    print(f"   Total tests: {total_tests}")
    print(f"   Successful H-CAF parses: {success_count}")
    print(f"   Success rate: {success_count/total_tests:.1%}")
    
    if success_count >= total_tests * 0.75:  # 75% success threshold
        print("🎉 JSON parsing fixes are working well!")
    elif success_count >= total_tests * 0.5:  # 50% success threshold
        print("⚡ JSON parsing has improved but may need further refinement")
    else:
        print("⚠️  JSON parsing fixes need more work")
    
    return success_count / total_tests


async def test_json_fix_function():
    """Test the _fix_json_string function specifically"""
    
    print("\n🔧 Testing JSON Fix Function")
    print("=" * 50)
    
    judge_config = PlatformConfig(
        name="test_judge",
        api_base="https://api.openai.com/v1", 
        model="gpt-4",
        api_key_env="OPENAI_API_KEY"
    )
    
    comparator = HCAFComparator(
        judge_config=judge_config,
        api_keys={"OPENAI_API_KEY": "test"},
        use_hcaf=True
    )
    
    test_cases = [
        # Trailing comma
        '{"key": "value",}',
        # Unquoted strings
        '{"verdict": SIGNIFICANT_DEGRADATION}',
        # Placeholders
        '{"score": <8>}',
        # Comments
        '{"key": "value", // comment\n"other": "value"}',
        # Complex case
        '{\n  "logical_reasoning": <7>,  // Good\n  "verdict": ON_PAR_OR_SUPERIOR,\n}'
    ]
    
    for i, broken_json in enumerate(test_cases, 1):
        print(f"\n🔧 Fix Test {i}:")
        print(f"   Input: {broken_json}")
        
        try:
            fixed = comparator._fix_json_string(broken_json)
            print(f"   Fixed: {fixed}")
            
            # Try to parse the fixed JSON
            parsed = json.loads(fixed)
            print(f"   ✅ Successfully parsed: {parsed}")
        except Exception as e:
            print(f"   ❌ Failed to fix/parse: {e}")


async def main():
    """Main test runner"""
    print("🚀 H-CAF JSON Parsing Test Suite")
    print("Testing enhanced JSON parsing capabilities for H-CAF framework")
    print("This helps ensure robust parsing of LLM responses with common JSON errors")
    
    # Run main parsing tests
    success_rate = await test_json_parsing()
    
    # Run specific fix function tests
    await test_json_fix_function()
    
    print("\n" + "=" * 60)
    print("🏁 ALL TESTS COMPLETED")
    
    if success_rate >= 0.75:
        print("✅ JSON parsing system is robust and ready for production")
    elif success_rate >= 0.5:
        print("⚡ JSON parsing has good coverage but monitor for edge cases")
    else:
        print("⚠️  Consider additional JSON parsing improvements")
    
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main()) 