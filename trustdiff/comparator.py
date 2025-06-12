"""
Comparator and Oracle logic for TrustDiff, including LLM-as-a-Judge functionality.
"""

import asyncio
from typing import Dict, Any, Optional
import httpx
from rich.console import Console

from .models import RawResult, EvaluationResult, QualityEvaluation, PlatformConfig, DetailedScores

console = Console()


class Comparator:
    """Comparator that acts as an oracle to evaluate differences between platforms."""
    
    def __init__(self, judge_config: PlatformConfig, api_keys: Dict[str, str], timeout_seconds: int = 60):
        self.judge_config = judge_config
        self.api_keys = api_keys
        self.timeout_seconds = timeout_seconds
        
        # Enhanced judge prompt template based on README.md specifications
        self.judge_prompt_template = """
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

# INSTRUCTIONS (Chain-of-Thought Process)
Follow these steps to generate your evaluation:
1. **Analyze Answer A**: Silently review Answer A and score it against each of the EVALUATION CRITERIA. Note its strengths and weaknesses.
2. **Analyze Answer B**: Silently review Answer B and score it against each of the EVALUATION CRITERIA. Note its strengths and weaknesses.
3. **Comparative Analysis**: Directly compare Answer A and Answer B. Note if the quality difference is minor (likely due to natural model variance) or significant (suggesting a difference in underlying model capability).
4. **Formulate Reasoning**: Based on your comparative analysis, write a concise but comprehensive explanation for your final verdict. This explanation MUST justify your final decision.
5. **Determine Final Verdict**: Based on the overall assessment, choose one of the three following verdict strings: `BASELINE_SUPERIOR`, `TARGET_SUPERIOR`, or `SIMILAR_QUALITY`. A verdict of `BASELINE_SUPERIOR` should be chosen only if Answer B is demonstrably and significantly better in one or more key criteria, beyond what could be attributed to random chance.
6. **Construct Final JSON**: Assemble all the above information into a single JSON object.

# INPUT
**User Prompt:**
{original_prompt}

**Answer A (Target Platform):**
{response_a}

**Answer B (Baseline):**
{response_b}

# OUTPUT FORMAT
Your entire output must be a single, valid JSON object, with no explanatory text before or after it. Use the following structure:

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
    
    def _extract_original_prompt(self, result: RawResult) -> str:
        """Extract the original prompt from the result context."""
        # Try to extract from the original request if available
        if result.response_data and 'messages' in str(result.response_data):
            # This is a simplified extraction - in a real implementation
            # you might want to store the original request alongside the response
            return "[Original prompt extracted from context]"
        return "[Original prompt not available - consider storing request data for better analysis]"
    
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
            "max_tokens": 1000,
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
                "maxOutputTokens": 2000,  # Increase token limit for complex evaluations
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
                    # Try to extract partial content
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
    
    def _parse_judge_response(self, response_data: Dict[str, Any]) -> Optional[QualityEvaluation]:
        """Parse the judge LLM response into a QualityEvaluation."""
        try:
            if 'choices' not in response_data or not response_data['choices']:
                return None
            
            content = response_data['choices'][0]['message']['content']
            
            # Try to extract JSON from the response
            import json
            import re
            
            # Look for JSON block
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    parsed = json.loads(json_str)
                    
                    # Parse new structured format
                    if 'scores' in parsed and 'final_verdict' in parsed:
                        # New detailed format
                        scores_data = parsed.get('scores', {})
                        answer_a_scores = scores_data.get('answer_a', {})
                        answer_b_scores = scores_data.get('answer_b', {})
                        
                        # Create detailed scores objects
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
                        
                        # Calculate average scores for backward compatibility
                        avg_target = sum(answer_a_scores.values()) / len(answer_a_scores) if answer_a_scores else 3.0
                        avg_baseline = sum(answer_b_scores.values()) / len(answer_b_scores) if answer_b_scores else 3.0
                        
                        return QualityEvaluation(
                            verdict=parsed.get('final_verdict', 'SIMILAR_QUALITY'),
                            confidence=0.8,  # High confidence for structured response
                            reasoning=parsed.get('comparative_reasoning', 'Detailed structured evaluation'),
                            comparative_reasoning=parsed.get('comparative_reasoning'),
                            detailed_scores_target=detailed_scores_target,
                            detailed_scores_baseline=detailed_scores_baseline,
                            score_target=avg_target,
                            score_baseline=avg_baseline
                        )
                    
                    # Legacy format support
                    elif 'verdict' in parsed:
                        return QualityEvaluation(
                            verdict=parsed.get('verdict', 'equivalent'),
                            confidence=float(parsed.get('confidence', 0.5)),
                            reasoning=parsed.get('reasoning', 'No reasoning provided'),
                            score_baseline=float(parsed.get('score_baseline')) if parsed.get('score_baseline') is not None else None,
                            score_target=float(parsed.get('score_target')) if parsed.get('score_target') is not None else None
                        )
                    
                except json.JSONDecodeError as e:
                    console.print(f"[yellow]JSON parsing failed: {e}[/yellow]")
                    pass
            
            # Fallback: try to extract information from text
            verdict = 'SIMILAR_QUALITY'
            if 'baseline_superior' in content.lower() or 'answer b' in content.lower():
                verdict = 'BASELINE_SUPERIOR'
            elif 'target_superior' in content.lower() or 'answer a' in content.lower():
                verdict = 'TARGET_SUPERIOR'
            elif 'baseline_better' in content.lower():
                verdict = 'baseline_better'
            elif 'target_better' in content.lower():
                verdict = 'target_better'
            elif 'both_poor' in content.lower():
                verdict = 'both_poor'
            
            return QualityEvaluation(
                verdict=verdict,
                confidence=0.3,  # Low confidence for text-based parsing
                reasoning=content[:500],  # Truncate long responses
            )
            
        except Exception as e:
            console.print(f"[yellow]Failed to parse judge response: {e}[/yellow]")
            return None
    
    def _get_probe_prompt_from_context(self, baseline: RawResult, target: RawResult) -> str:
        """Extract the original probe prompt by looking up the probe context."""
        # This is a more sophisticated approach that would ideally
        # access the original probe data. For now, we'll use a placeholder
        # but indicate how this could be improved.
        
        probe_id = baseline.probe_id
        
        # TODO: In a production implementation, you would:
        # 1. Store probe context with results
        # 2. Or maintain a probe registry to look up original prompts
        # 3. Or parse the original request from stored API call data
        
        return f"[Probe: {probe_id}] - Original prompt should be extracted from probe definition or stored request data"
    
    async def compare_quality_with_llm(self, baseline: RawResult, target: RawResult, original_probe_prompt: Optional[str] = None) -> Optional[QualityEvaluation]:
        """Compare quality using LLM-as-a-Judge."""
        # Extract response contents
        baseline_content = self._extract_response_content(baseline)
        target_content = self._extract_response_content(target)
        
        # Get original prompt - use provided prompt or extract from context
        if original_probe_prompt:
            original_prompt = original_probe_prompt
        else:
            original_prompt = self._get_probe_prompt_from_context(baseline, target)
        
        # Format the judge prompt correctly:
        # Answer A = Target platform (being tested)
        # Answer B = Baseline platform (ground truth)
        judge_prompt = self.judge_prompt_template.format(
            original_prompt=original_prompt,
            response_a=target_content,   # Target platform response (Answer A)
            response_b=baseline_content  # Baseline platform response (Answer B)
        )
        
        # Call judge LLM
        response_data = await self._call_judge_llm(judge_prompt)
        if not response_data:
            return None
        
        # Parse response
        evaluation = self._parse_judge_response(response_data)
        
        # Log the evaluation for debugging
        if evaluation:
            console.print(f"[dim]Judge evaluation for {baseline.probe_id}: {evaluation.verdict}[/dim]")
        
        return evaluation
    
    async def compare(self, baseline: RawResult, target: RawResult) -> EvaluationResult:
        """Perform complete comparison between baseline and target results."""
        try:
            # Basic performance metrics
            latency_diff = self.compare_latency(baseline, target)
            cost_diff = self.compare_cost(baseline, target)
            tokens_diff = self.compare_tokens(baseline, target)
            
            # Quality evaluation (if both results are successful)
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
            console.print(f"[red]Comparison failed for {baseline.probe_id}: {e}[/red]")
            return EvaluationResult(
                probe_id=baseline.probe_id,
                target_platform=target.platform_name,
                baseline_platform=baseline.platform_name,
                evaluation_success=False,
                error_message=str(e)
            )
    
    async def batch_compare(self, baseline_results: list[RawResult], target_results: list[RawResult]) -> list[EvaluationResult]:
        """Compare multiple result pairs concurrently."""
        if len(baseline_results) != len(target_results):
            raise ValueError("Baseline and target result lists must be the same length")
        
        tasks = []
        for baseline, target in zip(baseline_results, target_results):
            tasks.append(self.compare(baseline, target))
        
        return await asyncio.gather(*tasks, return_exceptions=True)


class SimpleComparator(Comparator):
    """Simplified comparator that doesn't use LLM-as-a-Judge."""
    
    def __init__(self):
        # Initialize without judge config
        self.judge_config = None
        self.api_keys = {}
    
    async def compare_quality_with_llm(self, baseline: RawResult, target: RawResult) -> Optional[QualityEvaluation]:
        """Simple quality comparison without LLM judge."""
        # Simple heuristic: compare response lengths
        baseline_content = self._extract_response_content(baseline)
        target_content = self._extract_response_content(target)
        
        baseline_len = len(baseline_content)
        target_len = len(target_content)
        
        if abs(baseline_len - target_len) < 50:  # Similar lengths
            verdict = "equivalent"
        elif target_len > baseline_len * 1.2:  # Target much longer
            verdict = "target_better"
        elif baseline_len > target_len * 1.2:  # Baseline much longer
            verdict = "baseline_better"
        else:
            verdict = "equivalent"
        
        return QualityEvaluation(
            verdict=verdict,
            confidence=0.3,  # Low confidence for simple heuristic
            reasoning=f"Simple comparison based on response length: baseline={baseline_len}, target={target_len}"
        ) 