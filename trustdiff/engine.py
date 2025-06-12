"""
Core testing engine for TrustDiff using asyncio and httpx.
"""

import os
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx
import yaml
from rich.console import Console
from rich.progress import Progress, TaskID

from .models import RunConfig, Probe, RawResult, EvaluationResult, TestSummary, ProbeMessage
from .storage import Storage
from .comparator import Comparator

console = Console()


class Engine:
    """Core testing engine that orchestrates the entire test process."""
    
    def __init__(self, config: RunConfig):
        self.config = config
        self.storage: Optional[Storage] = None
        self.comparator: Optional[Comparator] = None
        self.run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
    async def _initialize_run(self) -> str:
        """Initialize a new test run."""
        output_dir = Path(self.config.output_dir) / self.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.storage = Storage(str(output_dir))
        await self.storage.initialize()
        
        if self.config.judge:
            self.comparator = Comparator(self.config.judge, self.config.api_keys)
        
        return str(output_dir)
    
    def _load_probes(self, probe_filter: Optional[str] = None) -> List[Probe]:
        """Load probe definitions from the probe directory."""
        probe_dir = Path(self.config.probe_dir)
        if not probe_dir.exists():
            console.print(f"[red]Probe directory not found: {probe_dir}[/red]")
            return []
        
        probes = []
        for probe_file in probe_dir.rglob("*.yaml"):
            try:
                with open(probe_file, 'r', encoding='utf-8') as f:
                    probe_data = yaml.safe_load(f)
                
                # Convert prompt format
                if 'prompt' in probe_data:
                    if isinstance(probe_data['prompt'], list):
                        # Already in correct format
                        pass
                    elif isinstance(probe_data['prompt'], str):
                        # Convert string to message format
                        probe_data['prompt'] = [{"role": "user", "content": probe_data['prompt']}]
                    elif isinstance(probe_data['prompt'], dict):
                        # Convert single message dict to list
                        probe_data['prompt'] = [probe_data['prompt']]
                
                probe = Probe(**probe_data)
                
                # Apply filter if specified
                if probe_filter and probe_filter.lower() not in probe.probe_id.lower():
                    continue
                    
                probes.append(probe)
                
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to load probe {probe_file}: {e}[/yellow]")
        
        console.print(f"[blue]Loaded {len(probes)} probes[/blue]")
        return probes
    
    async def _make_api_request(
        self, 
        probe: Probe, 
        platform_config, 
        semaphore: asyncio.Semaphore
    ) -> RawResult:
        """Make an API request to a platform."""
        async with semaphore:
            start_time = datetime.now()
            
            try:
                # Get API key
                api_key = self.config.api_keys.get(platform_config.api_key_env, "")
                if not api_key:
                    console.print(f"[yellow]Warning: No API key found for {platform_config.name}[/yellow]")
                
                # Prepare headers
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                if platform_config.headers:
                    headers.update(platform_config.headers)
                
                # Prepare request body
                messages = []
                for msg in probe.prompt:
                    messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
                
                body = {
                    "model": platform_config.model or "gpt-3.5-turbo",
                    "messages": messages,
                    "max_tokens": probe.max_tokens or 1000,
                    "temperature": probe.temperature or 0.7
                }
                
                # Make request
                async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                    response = await client.post(
                        f"{platform_config.api_base}/chat/completions",
                        headers=headers,
                        json=body
                    )
                    
                    latency = (datetime.now() - start_time).total_seconds() * 1000
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        
                        # Extract token usage if available
                        tokens_used = None
                        if 'usage' in response_data:
                            tokens_used = response_data['usage'].get('total_tokens')
                        
                        return RawResult(
                            probe_id=probe.probe_id,
                            platform_name=platform_config.name,
                            success=True,
                            response_data=response_data,
                            latency_ms=latency,
                            tokens_used=tokens_used
                        )
                    else:
                        return RawResult(
                            probe_id=probe.probe_id,
                            platform_name=platform_config.name,
                            success=False,
                            error_message=f"HTTP {response.status_code}: {response.text}",
                            latency_ms=latency
                        )
                        
            except Exception as e:
                latency = (datetime.now() - start_time).total_seconds() * 1000
                return RawResult(
                    probe_id=probe.probe_id,
                    platform_name=platform_config.name,
                    success=False,
                    error_message=str(e),
                    latency_ms=latency
                )
    
    async def _run_all_requests(self, probes: List[Probe]) -> Dict[str, List[RawResult]]:
        """Run all API requests concurrently."""
        semaphore = asyncio.Semaphore(self.config.concurrency)
        tasks = []
        
        # Create tasks for all probe-platform combinations
        for probe in probes:
            # Baseline request
            tasks.append(self._make_api_request(probe, self.config.baseline, semaphore))
            
            # Target platform requests
            for target in self.config.targets:
                tasks.append(self._make_api_request(probe, target, semaphore))
        
        console.print(f"[blue]Running {len(tasks)} API requests with concurrency {self.config.concurrency}...[/blue]")
        
        # Execute all requests
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Group results by probe_id
        grouped_results: Dict[str, List[RawResult]] = {}
        for result in results:
            if isinstance(result, RawResult):
                if result.probe_id not in grouped_results:
                    grouped_results[result.probe_id] = []
                grouped_results[result.probe_id].append(result)
                
                # Save raw result
                if self.storage:
                    await self.storage.save_raw_result(result)
            else:
                console.print(f"[red]Request failed with exception: {result}[/red]")
        
        return grouped_results
    
    async def _evaluate_results(self, grouped_results: Dict[str, List[RawResult]]) -> List[EvaluationResult]:
        """Evaluate and compare results."""
        evaluations = []
        
        if not self.comparator:
            console.print("[yellow]No judge configured, skipping quality evaluation[/yellow]")
            # Return basic evaluations without quality assessment
            for probe_id, results in grouped_results.items():
                baseline_result = None
                target_results = []
                
                for result in results:
                    if result.platform_name == self.config.baseline.name:
                        baseline_result = result
                    else:
                        target_results.append(result)
                
                if baseline_result:
                    for target_result in target_results:
                        evaluation = EvaluationResult(
                            probe_id=probe_id,
                            target_platform=target_result.platform_name,
                            baseline_platform=baseline_result.platform_name,
                            latency_diff_ms=self._calc_latency_diff(baseline_result, target_result),
                            cost_diff=self._calc_cost_diff(baseline_result, target_result),
                            tokens_diff=self._calc_tokens_diff(baseline_result, target_result)
                        )
                        evaluations.append(evaluation)
            
            return evaluations
        
        # Full evaluation with quality assessment
        comparison_tasks = []
        for probe_id, results in grouped_results.items():
            baseline_result = None
            target_results = []
            
            for result in results:
                if result.platform_name == self.config.baseline.name:
                    baseline_result = result
                else:
                    target_results.append(result)
            
            if baseline_result:
                for target_result in target_results:
                    comparison_tasks.append(
                        self.comparator.compare(baseline_result, target_result)
                    )
        
        console.print(f"[blue]Running {len(comparison_tasks)} quality evaluations...[/blue]")
        evaluations = await asyncio.gather(*comparison_tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_evaluations = [
            eval_result for eval_result in evaluations 
            if isinstance(eval_result, EvaluationResult)
        ]
        
        # Save evaluations
        if self.storage:
            for evaluation in valid_evaluations:
                await self.storage.save_evaluation(evaluation)
        
        return valid_evaluations
    
    def _calc_latency_diff(self, baseline: RawResult, target: RawResult) -> Optional[float]:
        """Calculate latency difference."""
        if baseline.latency_ms is not None and target.latency_ms is not None:
            return target.latency_ms - baseline.latency_ms
        return None
    
    def _calc_cost_diff(self, baseline: RawResult, target: RawResult) -> Optional[float]:
        """Calculate cost difference."""
        if baseline.cost_estimate is not None and target.cost_estimate is not None:
            return target.cost_estimate - baseline.cost_estimate
        return None
    
    def _calc_tokens_diff(self, baseline: RawResult, target: RawResult) -> Optional[int]:
        """Calculate token usage difference."""
        if baseline.tokens_used is not None and target.tokens_used is not None:
            return target.tokens_used - baseline.tokens_used
        return None
    
    async def run_test_suite(self, probe_filter: Optional[str] = None) -> Optional[TestSummary]:
        """Run the complete test suite."""
        try:
            # Initialize
            output_dir = await self._initialize_run()
            
            # Load probes
            probes = self._load_probes(probe_filter)
            if not probes:
                console.print("[red]No probes found to run[/red]")
                return None
            
            # Run requests
            grouped_results = await self._run_all_requests(probes)
            
            # Evaluate results
            evaluations = await self._evaluate_results(grouped_results)
            
            # Generate summary
            total_evaluations = len(evaluations)
            successful_evaluations = sum(1 for e in evaluations if e.evaluation_success)
            success_rate = successful_evaluations / total_evaluations if total_evaluations > 0 else 0
            
            summary = TestSummary(
                run_id=self.run_id,
                timestamp=datetime.now(),
                total_probes=len(probes),
                total_platforms=len(self.config.targets),
                total_evaluations=total_evaluations,
                success_rate=success_rate,
                output_dir=output_dir
            )
            
            console.print(f"[green]✓ Test suite completed[/green]")
            console.print(f"  - Probes: {summary.total_probes}")
            console.print(f"  - Platforms: {summary.total_platforms}")
            console.print(f"  - Evaluations: {summary.total_evaluations}")
            console.print(f"  - Success rate: {summary.success_rate:.2%}")
            
            return summary
            
        except Exception as e:
            console.print(f"[red]Test suite failed: {e}[/red]")
            return None 