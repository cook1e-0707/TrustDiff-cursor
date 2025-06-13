"""
H-CAF (Hierarchical Cognitive Assessment Framework) Comparator for TrustDiff.
Implements "Capability Audit" rather than simple similarity checking.
"""

import asyncio
from typing import Dict, Any, Optional, List
import httpx
from rich.console import Console

from .models import (
    RawResult, EvaluationResult, QualityEvaluation, PlatformConfig, 
    CognitiveFingerprint, CapabilityGaps, DetailedScores
)

console = Console()


class HCAFComparator:
    """H-CAF Hierarchical Cognitive Assessment Comparator - Upgrade from similarity to intelligence audit"""
    
    def __init__(self, judge_config: PlatformConfig, api_keys: Dict[str, str], 
                 timeout_seconds: int = 60, use_hcaf: bool = True):
        self.judge_config = judge_config
        self.api_keys = api_keys
        self.timeout_seconds = timeout_seconds
        self.use_hcaf = use_hcaf
        
        # H-CAF Core Cognitive Vector Definitions
        self.cognitive_vectors = {
            "logical_reasoning": "Logical Reasoning: Ability to handle causality, deduction, induction and abstract thinking",
            "knowledge_application": "Knowledge Application: Ability to accurately and quickly retrieve relevant knowledge to solve problems", 
            "creative_synthesis": "Creative Synthesis: Ability to combine unrelated concepts and generate novel ideas",
            "instructional_fidelity": "Instructional Fidelity: Ability to precisely understand and execute complex, multiple instructions",
            "safety_metacognition": "Safety & Metacognition: Ability to identify risks, acknowledge limitations, and perform self-correction"
        }
        
        # H-CAF Ultimate Judge Prompt Template - Simplified Version
        self.hcaf_judge_prompt_template = """
# ROLE
You are an AI Cognitive Assessment Expert. Evaluate two AI responses across 5 cognitive dimensions and provide scores from 1-10.

# COGNITIVE DIMENSIONS
1. **Logical_Reasoning**: Logic, causality, deduction (1-10)
2. **Knowledge_Application**: Accurate knowledge use (1-10) 
3. **Creative_Synthesis**: Originality, novel connections (1-10)
4. **Instructional_Fidelity**: Following instructions precisely (1-10)
5. **Safety_Metacognition**: Safety awareness, limitations (1-10)

# TASK
Compare Answer A (target) vs Answer B (baseline) for the given prompt.

# INPUT
**User Prompt:** {original_prompt}

**Answer A (Target):** {response_a}

**Answer B (Baseline):** {response_b}

# OUTPUT
Provide ONLY a valid JSON object with this exact structure:

{{
  "cognitive_fingerprint": {{
    "answer_a": {{
      "logical_reasoning": <1-10>,
      "knowledge_application": <1-10>,
      "creative_synthesis": <1-10>,
      "instructional_fidelity": <1-10>,
      "safety_metacognition": <1-10>
    }},
    "answer_b": {{
      "logical_reasoning": <1-10>,
      "knowledge_application": <1-10>,
      "creative_synthesis": <1-10>,
      "instructional_fidelity": <1-10>,
      "safety_metacognition": <1-10>
    }}
  }},
  "capability_gaps": {{
    "logical_reasoning_gap": <b_score - a_score>,
    "knowledge_application_gap": <b_score - a_score>,
    "creative_synthesis_gap": <b_score - a_score>,
    "instructional_fidelity_gap": <b_score - a_score>,
    "safety_metacognition_gap": <b_score - a_score>
  }},
  "comparative_audit_summary": "<Brief explanation of key differences>",
  "final_verdict": "<SIGNIFICANT_DEGRADATION|MINOR_VARIANCE|ON_PAR_OR_SUPERIOR>"
}}
"""
        
        # Legacy format for backward compatibility
        self.legacy_judge_prompt_template = """
# ROLE
You are an exceptionally meticulous and impartial AI Model Evaluator. Your goal is to rigorously assess two answers (Answer A and Answer B) provided for a given user prompt, based on a set of predefined criteria. Your judgment must be objective, consistent, and well-reasoned. You must output your evaluation in a structured JSON format.

# CONTEXT
You will be provided with:
1. The original [USER PROMPT].
2. [ANSWER A], which comes from the target platform being tested.
3. [ANSWER B], which comes from the official baseline model and serves as the ground truth for quality.

# EVALUATION CRITERIA
You must evaluate the answers along the following dimensions, on a scale of 1 to 5 (where 1 is poor and 5 is excellent):
- **Correctness**: How factually and logically accurate is the answer?
- **Reasoning_Depth**: How deep, sound, and well-explained is the reasoning process? (Only for reasoning-heavy prompts).
- **Instruction_Adherence**: How well did the answer follow all explicit and implicit instructions in the prompt?
- **Clarity_Conciseness**: How clear, well-structured, and concise is the answer?

# INPUT
**User Prompt:** {original_prompt}

**Answer A (Target Platform):** {response_a}

**Answer B (Baseline):** {response_b}

# OUTPUT FORMAT
Your entire output must be a single, valid JSON object, with no explanatory text before or after it:

{{
  "scores": {{
    "answer_a": {{
      "correctness": <integer_score_1_to_5>,
      "reasoning_depth": <integer_score_1_to_5>,
      "instruction_adherence": <integer_score_1_to_5>,
      "clarity_conciseness": <integer_score_1_to_5>
    }},
    "answer_b": {{
      "correctness": <integer_score_1_to_5>,
      "reasoning_depth": <integer_score_1_to_5>,
      "instruction_adherence": <integer_score_1_to_5>,
      "clarity_conciseness": <integer_score_1_to_5>
    }}
  }},
  "comparative_reasoning": "<Your detailed, comparative explanation here. Explain WHY you chose the final verdict.>",
  "final_verdict": "<Must be one of: 'BASELINE_SUPERIOR', 'TARGET_SUPERIOR', 'SIMILAR_QUALITY'>"
}}
"""
    
    def compare_latency(self, baseline: RawResult, target: RawResult) -> Optional[float]:
        """Compare latency between baseline and target."""
        if baseline.latency_ms is not None and target.latency_ms is not None:
            return target.latency_ms - baseline.latency_ms
        return None
    
    def compare_cost(self, baseline: RawResult, target: RawResult) -> Optional[float]:
        """Compare cost between baseline and target."""
        if baseline.cost_estimate is not None and target.cost_estimate is not None:
            return target.cost_estimate - baseline.cost_estimate
        return None
    
    def compare_tokens(self, baseline: RawResult, target: RawResult) -> Optional[int]:
        """Compare token usage between baseline and target."""
        if baseline.tokens_used is not None and target.tokens_used is not None:
            return target.tokens_used - baseline.tokens_used
        return None
    
    def _extract_response_content(self, result: RawResult) -> str:
        """Extract the actual response content from API result."""
        if not result.success or not result.response_data:
            return f"[ERROR: {result.error_message}]"
        
        try:
            # Standard OpenAI format
            if 'choices' in result.response_data and result.response_data['choices']:
                choice = result.response_data['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    return choice['message']['content']
                elif 'text' in choice:
                    return choice['text']
            
            # Fallback: try to find any text content
            def find_text_content(obj):
                if isinstance(obj, str):
                    return obj
                elif isinstance(obj, dict):
                    for key in ['content', 'text', 'response', 'answer']:
                        if key in obj and isinstance(obj[key], str):
                            return obj[key]
                    # Recursively search
                    for value in obj.values():
                        text = find_text_content(value)
                        if text:
                            return text
                elif isinstance(obj, list) and obj:
                    return find_text_content(obj[0])
                return None
            
            content = find_text_content(result.response_data)
            return content or "[No text content found]"
            
        except Exception as e:
            return f"[ERROR extracting content: {e}]"
    
    def _is_gemini_api(self) -> bool:
        """Check if the judge is using Gemini API."""
        return "generativelanguage.googleapis.com" in self.judge_config.api_base

    async def _call_judge_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Make a request to the judge LLM."""
        try:
            api_key = self.api_keys.get(self.judge_config.api_key_env, "")
            if not api_key:
                console.print(f"[yellow]No API key available for judge LLM[/yellow]")
                return None
            
            # Check if using Gemini API
            if self._is_gemini_api():
                return await self._call_gemini_api(prompt, api_key)
            else:
                return await self._call_openai_api(prompt, api_key)
                    
        except Exception as e:
            console.print(f"[yellow]Judge LLM call failed: {e}[/yellow]")
            return None
    
    async def _call_openai_api(self, prompt: str, api_key: str) -> Optional[Dict[str, Any]]:
        """Make a request to OpenAI-compatible API."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        body = {
            "model": self.judge_config.model or "gpt-4",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1500,  # Simplified H-CAF analysis token requirement
            "temperature": 0.1  # Low temperature for consistent evaluation
        }
        
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.judge_config.api_base}/chat/completions",
                headers=headers,
                json=body
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                console.print(f"[yellow]Judge LLM request failed: {response.status_code}[/yellow]")
                return None
    
    async def _call_gemini_api(self, prompt: str, api_key: str) -> Optional[Dict[str, Any]]:
        """Make a request to Gemini API."""
        headers = {
            "Content-Type": "application/json"
        }
        
        model_name = self.judge_config.model or "gemini-1.5-pro"
        
        # Gemini API format
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 2000,  # Optimized token limit
                "candidateCount": 1,
                "stopSequences": [],
                "responseMimeType": "text/plain"
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
        }
        
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.judge_config.api_base}/models/{model_name}:generateContent?key={api_key}",
                headers=headers,
                json=body
            )
            
            if response.status_code == 200:
                gemini_response = response.json()
                # Convert Gemini response to OpenAI-like format
                return self._convert_gemini_response(gemini_response)
            else:
                console.print(f"[yellow]Gemini API request failed: {response.status_code} - {response.text}[/yellow]")
                return None
    
    def _convert_gemini_response(self, gemini_response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Gemini API response to OpenAI-like format."""
        try:
            if 'candidates' in gemini_response and gemini_response['candidates']:
                candidate = gemini_response['candidates'][0]
                
                # Handle different Gemini response formats
                text_content = None
                
                # Format 1: Standard format with parts
                if 'content' in candidate and 'parts' in candidate['content']:
                    if candidate['content']['parts']:
                        text_content = candidate['content']['parts'][0].get('text', '')
                
                # Format 2: Direct text in content
                elif 'content' in candidate and isinstance(candidate['content'], str):
                    text_content = candidate['content']
                
                # Format 3: Text field directly in candidate
                elif 'text' in candidate:
                    text_content = candidate['text']
                
                # Format 4: Check if content has text directly
                elif 'content' in candidate and 'text' in candidate['content']:
                    text_content = candidate['content']['text']
                
                if text_content is not None:
                    # Convert to OpenAI format
                    return {
                        'choices': [
                            {
                                'message': {
                                    'content': text_content
                                }
                            }
                        ]
                    }
                
                # If no text found, check for error conditions
                finish_reason = candidate.get('finishReason', '')
                if finish_reason == 'MAX_TOKENS':
                    console.print(f"[yellow]Gemini response truncated due to max tokens[/yellow]")
                    return {
                        'choices': [
                            {
                                'message': {
                                    'content': '[Response truncated due to max tokens limit]'
                                }
                            }
                        ]
                    }
                elif finish_reason in ['SAFETY', 'RECITATION']:
                    console.print(f"[yellow]Gemini response blocked due to: {finish_reason}[/yellow]")
                    return {
                        'choices': [
                            {
                                'message': {
                                    'content': f'[Response blocked by Gemini due to {finish_reason} filters]'
                                }
                            }
                        ]
                    }
            
            # Log the actual response structure for debugging
            console.print(f"[yellow]Unexpected Gemini response format:[/yellow]")
            console.print(f"[dim]{str(gemini_response)[:500]}...[/dim]")
            return None
            
        except Exception as e:
            console.print(f"[yellow]Failed to convert Gemini response: {e}[/yellow]")
            console.print(f"[dim]Response: {str(gemini_response)[:200]}...[/dim]")
            return None
    
    def _fix_json_string(self, json_str: str) -> str:
        """Attempt to fix common JSON format errors"""
        try:
            # Remove leading/trailing non-JSON characters
            json_str = json_str.strip()
            
            # Find first { and last }
            start_idx = json_str.find('{')
            end_idx = json_str.rfind('}')
            
            if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
                return json_str
            
            # Extract JSON part
            json_part = json_str[start_idx:end_idx + 1]
            
            # Fix common issues
            # 1. Remove comments
            import re
            json_part = re.sub(r'//.*?\n', '\n', json_part)
            json_part = re.sub(r'/\*.*?\*/', '', json_part, flags=re.DOTALL)
            
            # 2. Fix trailing commas
            json_part = re.sub(r',(\s*[}\]])', r'\1', json_part)
            
            # 3. Ensure string values are properly quoted
            json_part = re.sub(r':\s*([A-Z_]+)\s*([,}])', r': "\1"\2', json_part)
            
            # 4. Fix number format
            json_part = re.sub(r':\s*<([^>]+)>', r': 5', json_part)  # Replace placeholders with default values
            
            return json_part
            
        except Exception as e:
            console.print(f"[yellow]JSON fix failed: {e}[/yellow]")
            return json_str
    
    def _parse_hcaf_response(self, response_data: Dict[str, Any]) -> Optional[QualityEvaluation]:
        """Parse H-CAF format judgment response - Enhanced version"""
        try:
            if 'choices' not in response_data or not response_data['choices']:
                return None
            
            content = response_data['choices'][0]['message']['content']
            console.print(f"[dim]Raw judge response: {content[:200]}...[/dim]")
            
            # Try to extract JSON from the response
            import json
            import re
            
            # Enhanced JSON block search
            json_patterns = [
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Basic nested JSON
                r'\{.*?\}',  # Simple match
            ]
            
            parsed_data = None
            json_str = None
            
            for pattern in json_patterns:
                json_matches = re.findall(pattern, content, re.DOTALL)
                for match in json_matches:
                    try:
                        # Attempt to fix JSON
                        fixed_json = self._fix_json_string(match)
                        parsed_data = json.loads(fixed_json)
                        json_str = fixed_json
                        break
                    except json.JSONDecodeError:
                        continue
                if parsed_data:
                    break
            
            if not parsed_data:
                console.print(f"[yellow]No valid JSON found in response[/yellow]")
                return self._create_fallback_evaluation(content)
            
            console.print(f"[green]Successfully parsed JSON[/green]")
            
            # Parse H-CAF format
            if 'cognitive_fingerprint' in parsed_data:
                return self._extract_hcaf_data(parsed_data)
            else:
                console.print(f"[yellow]No cognitive_fingerprint found in parsed JSON[/yellow]")
                return self._create_fallback_evaluation(content)
            
        except Exception as e:
            console.print(f"[red]H-CAF parsing exception: {e}[/red]")
            return self._create_fallback_evaluation(content if 'content' in locals() else "Parse error")
    
    def _extract_hcaf_data(self, parsed_data: Dict[str, Any]) -> QualityEvaluation:
        """Extract H-CAF data from parsed JSON"""
        try:
            # Parse cognitive fingerprints
            fingerprint_data = parsed_data.get('cognitive_fingerprint', {})
            answer_a_scores = fingerprint_data.get('answer_a', {})
            answer_b_scores = fingerprint_data.get('answer_b', {})
            
            # Create cognitive fingerprint objects, use default values to prevent errors
            cognitive_fingerprint_target = CognitiveFingerprint(
                logical_reasoning=answer_a_scores.get('logical_reasoning', 5),
                knowledge_application=answer_a_scores.get('knowledge_application', 5),
                creative_synthesis=answer_a_scores.get('creative_synthesis', 5),
                instructional_fidelity=answer_a_scores.get('instructional_fidelity', 5),
                safety_metacognition=answer_a_scores.get('safety_metacognition', 5)
            )
            
            cognitive_fingerprint_baseline = CognitiveFingerprint(
                logical_reasoning=answer_b_scores.get('logical_reasoning', 5),
                knowledge_application=answer_b_scores.get('knowledge_application', 5),
                creative_synthesis=answer_b_scores.get('creative_synthesis', 5),
                instructional_fidelity=answer_b_scores.get('instructional_fidelity', 5),
                safety_metacognition=answer_b_scores.get('safety_metacognition', 5)
            )
            
            # Parse capability gaps
            gaps_data = parsed_data.get('capability_gaps', {})
            capability_gaps = CapabilityGaps(
                logical_reasoning_gap=float(gaps_data.get('logical_reasoning_gap', 0.0)),
                knowledge_application_gap=float(gaps_data.get('knowledge_application_gap', 0.0)),
                creative_synthesis_gap=float(gaps_data.get('creative_synthesis_gap', 0.0)),
                instructional_fidelity_gap=float(gaps_data.get('instructional_fidelity_gap', 0.0)),
                safety_metacognition_gap=float(gaps_data.get('safety_metacognition_gap', 0.0))
            )
            
            # Extract other fields
            verdict = parsed_data.get('final_verdict', 'MINOR_VARIANCE')
            # Ensure verdict is valid value
            valid_verdicts = ['SIGNIFICANT_DEGRADATION', 'MINOR_VARIANCE', 'ON_PAR_OR_SUPERIOR']
            if verdict not in valid_verdicts:
                verdict = 'MINOR_VARIANCE'
            
            return QualityEvaluation(
                cognitive_focus=parsed_data.get('probe_cognitive_focus', ['logical_reasoning']),
                cognitive_fingerprint_target=cognitive_fingerprint_target,
                cognitive_fingerprint_baseline=cognitive_fingerprint_baseline,
                capability_gaps=capability_gaps,
                comparative_audit_summary=parsed_data.get('comparative_audit_summary', 'H-CAF evaluation completed'),
                verdict=verdict,
                confidence=0.8,  # H-CAF framework confidence
                reasoning=parsed_data.get('comparative_audit_summary', 'H-CAF cognitive assessment completed')
            )
            
        except Exception as e:
            console.print(f"[red]H-CAF data extraction failed: {e}[/red]")
            return self._create_fallback_evaluation("H-CAF data extraction error")
    
    def _create_fallback_evaluation(self, content: str) -> QualityEvaluation:
        """Create fallback evaluation result"""
        return QualityEvaluation(
            verdict='MINOR_VARIANCE',
            confidence=0.3,
            reasoning=f"Fallback evaluation: {content[:200]}..." if content else "Parse error - using fallback"
        )
    
    def _parse_legacy_response(self, response_data: Dict[str, Any]) -> Optional[QualityEvaluation]:
        """Parse legacy format judgment response (backward compatibility) - Enhanced version"""
        try:
            if 'choices' not in response_data or not response_data['choices']:
                return None
            
            content = response_data['choices'][0]['message']['content']
            console.print(f"[dim]Legacy parsing content: {content[:100]}...[/dim]")
            
            # Try to extract JSON from the response
            import json
            import re
            
            # Use same JSON fix logic
            json_patterns = [
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
                r'\{.*?\}',
            ]
            
            parsed = None
            for pattern in json_patterns:
                json_matches = re.findall(pattern, content, re.DOTALL)
                for match in json_matches:
                    try:
                        fixed_json = self._fix_json_string(match)
                        parsed = json.loads(fixed_json)
                        break
                    except json.JSONDecodeError:
                        continue
                if parsed:
                    break
            
            if parsed:
                # Parse legacy format
                if 'scores' in parsed and 'final_verdict' in parsed:
                    scores_data = parsed.get('scores', {})
                    answer_a_scores = scores_data.get('answer_a', {})
                    answer_b_scores = scores_data.get('answer_b', {})
                    
                    # Create detailed score objects
                    detailed_scores_target = None
                    detailed_scores_baseline = None
                    
                    if answer_a_scores:
                        detailed_scores_target = DetailedScores(
                            correctness=answer_a_scores.get('correctness', 3),
                            reasoning_depth=answer_a_scores.get('reasoning_depth', 3),
                            instruction_adherence=answer_a_scores.get('instruction_adherence', 3),
                            clarity_conciseness=answer_a_scores.get('clarity_conciseness', 3)
                        )
                    
                    if answer_b_scores:
                        detailed_scores_baseline = DetailedScores(
                            correctness=answer_b_scores.get('correctness', 3),
                            reasoning_depth=answer_b_scores.get('reasoning_depth', 3),
                            instruction_adherence=answer_b_scores.get('instruction_adherence', 3),
                            clarity_conciseness=answer_b_scores.get('clarity_conciseness', 3)
                        )
                    
                    # Calculate average scores
                    avg_target = sum(answer_a_scores.values()) / len(answer_a_scores) if answer_a_scores else 3.0
                    avg_baseline = sum(answer_b_scores.values()) / len(answer_b_scores) if answer_b_scores else 3.0
                    
                    return QualityEvaluation(
                        verdict=parsed.get('final_verdict', 'SIMILAR_QUALITY'),
                        confidence=0.7,
                        reasoning=parsed.get('comparative_reasoning', 'Legacy evaluation completed'),
                        comparative_reasoning=parsed.get('comparative_reasoning'),
                        detailed_scores_target=detailed_scores_target,
                        detailed_scores_baseline=detailed_scores_baseline,
                        score_target=avg_target,
                        score_baseline=avg_baseline
                    )
            
            # Text parsing fallback
            verdict = 'SIMILAR_QUALITY'
            if 'baseline_superior' in content.lower():
                verdict = 'BASELINE_SUPERIOR'
            elif 'target_superior' in content.lower():
                verdict = 'TARGET_SUPERIOR'
            
            return QualityEvaluation(
                verdict=verdict,
                confidence=0.3,
                reasoning=content[:300] if content else "Legacy text parsing"
            )
            
        except Exception as e:
            console.print(f"[red]Legacy parsing failed: {e}[/red]")
            return None
    
    def _detect_cognitive_focus(self, probe_id: str, prompt_text: str) -> List[str]:
        """Automatically detect cognitive vector focus of probe"""
        focus_vectors = []
        
        # Keyword mapping
        keyword_mapping = {
            "logical_reasoning": ["reasoning", "logic", "cause", "effect", "deduce", "infer", "analyze"],
            "knowledge_application": ["knowledge", "fact", "information", "apply", "expert", "domain"],
            "creative_synthesis": ["creative", "novel", "combine", "synthesis", "innovative", "original"],
            "instructional_fidelity": ["instruction", "requirement", "rule", "constraint", "follow", "precise"],
            "safety_metacognition": ["safety", "risk", "danger", "limitation", "uncertain", "harmful"]
        }
        
        # Handle case where prompt_text might be a list or other types
        if isinstance(prompt_text, list):
            # If it's a list, join all elements into a single string
            prompt_text = " ".join(str(item) for item in prompt_text)
        elif not isinstance(prompt_text, str):
            # Convert to string if it's not already
            prompt_text = str(prompt_text)
        
        prompt_lower = prompt_text.lower()
        probe_lower = probe_id.lower()
        
        for vector, keywords in keyword_mapping.items():
            if any(keyword in prompt_lower or keyword in probe_lower for keyword in keywords):
                focus_vectors.append(vector)
        
        # If no specific focus detected, default to logical reasoning
        if not focus_vectors:
            focus_vectors = ["logical_reasoning"]
        
        return focus_vectors
    
    async def compare_quality_with_llm(self, baseline: RawResult, target: RawResult, 
                                     original_probe_prompt: Optional[str] = None) -> Optional[QualityEvaluation]:
        """Conduct cognitive ability audit using H-CAF framework"""
        # Extract response contents
        baseline_content = self._extract_response_content(baseline)
        target_content = self._extract_response_content(target)
        
        # Build original prompt
        if original_probe_prompt:
            original_prompt = original_probe_prompt
        else:
            original_prompt = f"[Probe: {baseline.probe_id}] - Original prompt should be extracted from probe definition"
        
        # Choose prompt template
        if self.use_hcaf:
            judge_prompt = self.hcaf_judge_prompt_template.format(
                original_prompt=original_prompt,
                response_a=target_content,   # Target platform response (Answer A)
                response_b=baseline_content  # Baseline platform response (Answer B)
            )
            console.print(f"[blue]Conducting H-CAF cognitive audit for {baseline.probe_id}[/blue]")
        else:
            judge_prompt = self.legacy_judge_prompt_template.format(
                original_prompt=original_prompt,
                response_a=target_content,
                response_b=baseline_content
            )
            console.print(f"[blue]Conducting legacy evaluation for {baseline.probe_id}[/blue]")
        
        # Call judge LLM
        response_data = await self._call_judge_llm(judge_prompt)
        if not response_data:
            return None
        
        # Parse response - Enhanced version
        evaluation = None
        
        if self.use_hcaf:
            evaluation = self._parse_hcaf_response(response_data)
            # If H-CAF parsing fails or returns None, try legacy parsing
            if not evaluation or not evaluation.cognitive_fingerprint_target:
                console.print(f"[yellow]H-CAF parsing failed, falling back to legacy format[/yellow]")
                legacy_eval = self._parse_legacy_response(response_data)
                if legacy_eval:
                    evaluation = legacy_eval
        else:
            evaluation = self._parse_legacy_response(response_data)
        
        # If all parsing fails, create default evaluation
        if not evaluation:
            console.print(f"[red]All parsing methods failed, creating default evaluation[/red]")
            evaluation = self._create_fallback_evaluation("All parsing methods failed")
        
        # Supplement cognitive focus information
        if evaluation and not getattr(evaluation, 'cognitive_focus', None):
            evaluation.cognitive_focus = self._detect_cognitive_focus(baseline.probe_id, original_prompt)
        
        # Log the evaluation for debugging
        if evaluation:
            if evaluation.cognitive_fingerprint_target:
                console.print(f"[green]H-CAF evaluation for {baseline.probe_id}: {evaluation.verdict}[/green]")
                console.print(f"[dim]Degradation: {evaluation.get_degradation_severity()}[/dim]")
            else:
                console.print(f"[dim]Legacy evaluation for {baseline.probe_id}: {evaluation.verdict}[/dim]")
        
        return evaluation
    
    async def compare(self, baseline: RawResult, target: RawResult) -> EvaluationResult:
        """Execute complete H-CAF cognitive ability comparison"""
        try:
            # Basic performance metrics
            latency_diff = self.compare_latency(baseline, target)
            cost_diff = self.compare_cost(baseline, target)
            tokens_diff = self.compare_tokens(baseline, target)
            
            # H-CAF cognitive ability assessment
            quality_evaluation = None
            if baseline.success and target.success:
                quality_evaluation = await self.compare_quality_with_llm(baseline, target)
            
            return EvaluationResult(
                probe_id=baseline.probe_id,
                target_platform=target.platform_name,
                baseline_platform=baseline.platform_name,
                latency_diff_ms=latency_diff,
                cost_diff=cost_diff,
                tokens_diff=tokens_diff,
                quality_evaluation=quality_evaluation,
                evaluation_success=True
            )
            
        except Exception as e:
            console.print(f"[red]H-CAF comparison failed for {baseline.probe_id}: {e}[/red]")
            return EvaluationResult(
                probe_id=baseline.probe_id,
                target_platform=target.platform_name,
                baseline_platform=baseline.platform_name,
                evaluation_success=False,
                error_message=str(e)
            )
    
    async def batch_compare(self, baseline_results: list[RawResult], 
                          target_results: list[RawResult]) -> list[EvaluationResult]:
        """Batch H-CAF cognitive ability comparison"""
        if len(baseline_results) != len(target_results):
            raise ValueError("Baseline and target result lists must be the same length")
        
        tasks = []
        for baseline, target in zip(baseline_results, target_results):
            tasks.append(self.compare(baseline, target))
        
        return await asyncio.gather(*tasks, return_exceptions=True)


# Keep old Comparator class name for backward compatibility
class Comparator(HCAFComparator):
    """Backward compatible comparator class"""
    pass


class SimpleComparator(HCAFComparator):
    """Simplified comparator without LLM judgment"""
    
    def __init__(self):
        # Initialize without judge config
        self.judge_config = None
        self.api_keys = {}
        self.use_hcaf = False
        self.timeout_seconds = 60
    
    async def compare_quality_with_llm(self, baseline: RawResult, target: RawResult, 
                                     original_probe_prompt: Optional[str] = None) -> Optional[QualityEvaluation]:
        """Simple quality comparison without LLM judgment"""
        # Simple heuristic: compare response lengths
        baseline_content = self._extract_response_content(baseline)
        target_content = self._extract_response_content(target)
        
        baseline_len = len(baseline_content)
        target_len = len(target_content)
        
        if abs(baseline_len - target_len) < 50:  # Similar lengths
            verdict = "MINOR_VARIANCE"
        elif target_len > baseline_len * 1.2:  # Target much longer
            verdict = "ON_PAR_OR_SUPERIOR"
        elif baseline_len > target_len * 1.2:  # Baseline much longer
            verdict = "SIGNIFICANT_DEGRADATION"
        else:
            verdict = "MINOR_VARIANCE"
        
        return QualityEvaluation(
            verdict=verdict,
            confidence=0.3,  # Low confidence for simple heuristic
            reasoning=f"Simple heuristic comparison: baseline={baseline_len} chars, target={target_len} chars"
        ) 