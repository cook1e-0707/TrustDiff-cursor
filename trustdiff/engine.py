"""
Core evaluation engine for the TrustDiff H-CAF Framework.
Orchestrates probe execution, platform comparison, and result aggregation.
"""

import asyncio
import os
import time
import httpx
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from .models import (
    RunConfig, ProbeDefinition, RawResult, EvaluationResult, TrustDiffReport, 
    ExecutionPlan, PlatformConfig
)
from .storage import TrustDiffStorage
from .comparator import HCAFComparator

console = Console()


class TrustDiffEngine:
    """
    Orchestrates the TrustDiff evaluation process, from loading probes
    to running comparisons and generating a final report.
    """
    
    def __init__(self, config: Dict[str, Any], run_config: RunConfig, storage: TrustDiffStorage):
        self.config = config
        self.run_config = run_config
        self.storage = storage
        self.api_keys = self._load_api_keys()
        
        # Initialize platforms
        self.target_platform = self._create_platform_config('target')
        self.baseline_platform = self._create_platform_config('baseline')
        self.judge_platform = self._create_platform_config('judge') if 'judge' in config else None
        
        # Initialize execution plan
        self.execution_plan = ExecutionPlan(
            probes=[],
            target_platform=self.target_platform,
            baseline_platform=self.baseline_platform,
            judge_platform=self.judge_platform,
            run_config=self.run_config
        )
        
        # Initialize comparator
        self.comparator = HCAFComparator(
            judge_config=self.judge_platform,
            api_keys=self.api_keys,
            timeout_seconds=self.run_config.timeout_seconds,
            use_hcaf=self.run_config.use_hcaf_framework
        ) if self.judge_platform else None

    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables."""
        api_keys = {}
        # Scan through platform configs to find all required API key env vars
        for platform_type in ['target', 'baseline', 'judge']:
            if platform_type in self.config:
                key_env = self.config[platform_type].get('api_key_env')
                if key_env and key_env not in api_keys:
                    api_keys[key_env] = os.getenv(key_env, "")
        return api_keys
    
    def _create_platform_config(self, platform_type: str) -> PlatformConfig:
        """Create a PlatformConfig object from the main configuration."""
        p_config = self.config.get(platform_type, {})
        return PlatformConfig(
            name=p_config.get('name', platform_type),
            api_base=p_config.get('api_base', ''),
            model=p_config.get('model', ''),
            api_key_env=p_config.get('api_key_env', ''),
            max_tokens=p_config.get('max_tokens', 1000),
            temperature=p_config.get('temperature', 0.7),
            additional_params=p_config.get('additional_params')
        )

    async def load_probes_from_config(self):
        """Load all probe definitions from directories specified in the config."""
        probe_dirs = self.config.get('probes', {}).get('directories', ['probes/'])
        probe_patterns = self.config.get('probes', {}).get('patterns', ['*.yaml', '*.yml'])
        
        all_probes = []
        for directory in probe_dirs:
            p_dir = Path(directory)
            if not p_dir.exists():
                console.print(f"[yellow]Warning: Probe directory not found: {p_dir}[/yellow]")
                continue
            
            for pattern in probe_patterns:
                for probe_file in p_dir.rglob(pattern):
                    try:
                        with open(probe_file, 'r', encoding='utf-8') as f:
                            probe_data = yaml.safe_load(f)
                        
                        probe = ProbeDefinition(**probe_data)
                        all_probes.append(probe)
                    except Exception as e:
                        console.print(f"[red]Failed to load probe {probe_file}: {e}[/red]")
                        
        self.execution_plan.probes = all_probes

    async def _send_request(self, probe: ProbeDefinition, platform: PlatformConfig, semaphore: asyncio.Semaphore) -> RawResult:
        """Send a single API request to a platform."""
        async with semaphore:
            start_time = time.monotonic()
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_keys.get(platform.api_key_env, '')}"
            }
            
            body = {
                "model": platform.model,
                "messages": [{"role": "user", "content": probe.prompt}],
                "max_tokens": platform.max_tokens,
                "temperature": platform.temperature,
                **(platform.additional_params or {})
            }

            try:
                async with httpx.AsyncClient(timeout=self.run_config.timeout_seconds) as client:
                    response = await client.post(
                        f"{platform.api_base.rstrip('/')}/chat/completions",
                        headers=headers,
                        json=body
                    )
                    
                    latency_ms = (time.monotonic() - start_time) * 1000
                    
                    if response.status_code == 200:
                        return RawResult(
                            probe_id=probe.id,
                            platform_name=platform.name,
                            success=True,
                            response_data=response.json(),
                            latency_ms=latency_ms,
                            timestamp=datetime.now()
                        )
                    else:
                        return RawResult(
                            probe_id=probe.id,
                            platform_name=platform.name,
                            success=False,
                            error_message=f"HTTP {response.status_code}: {response.text[:200]}",
                            latency_ms=latency_ms,
                            timestamp=datetime.now()
                        )
            except Exception as e:
                latency_ms = (time.monotonic() - start_time) * 1000
                return RawResult(
                    probe_id=probe.id,
                    platform_name=platform.name,
                    success=False,
                    error_message=str(e),
                    latency_ms=latency_ms,
                    timestamp=datetime.now()
                )

    async def _run_probe_on_platforms(self, probe: ProbeDefinition, progress: Progress, task: TaskID) -> (RawResult, RawResult):
        """Run a single probe on both baseline and target platforms."""
        semaphore = asyncio.Semaphore(2) # For a single probe, max 2 concurrent requests
        
        baseline_task = self._send_request(probe, self.baseline_platform, semaphore)
        target_task = self._send_request(probe, self.target_platform, semaphore)
        
        baseline_result, target_result = await asyncio.gather(baseline_task, target_task)
        
        progress.update(task, advance=1)
        return baseline_result, target_result

    async def run_complete_evaluation(self) -> TrustDiffReport:
        """
        Execute the full evaluation suite: run all probes, compare results,
        and generate a comprehensive report.
        """
        start_time = time.monotonic()
        
        raw_results_baseline: List[RawResult] = []
        raw_results_target: List[RawResult] = []
        evaluation_results: List[EvaluationResult] = []
        
        total_probes = len(self.execution_plan.probes)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            transient=True
        ) as progress:
            # Phase 1: Run probes
            probe_task = progress.add_task("[cyan]Running Probes...", total=total_probes)
            probe_run_tasks = [self._run_probe_on_platforms(probe, progress, probe_task) for probe in self.execution_plan.probes]
            
            probe_results = await asyncio.gather(*probe_run_tasks)
            
            for baseline_res, target_res in probe_results:
                raw_results_baseline.append(baseline_res)
                raw_results_target.append(target_res)

            # Phase 2: Run comparisons
            if self.comparator:
                compare_task = progress.add_task("[magenta]Comparing Results...", total=total_probes)
                
                comparison_tasks = []
                for baseline_res, target_res in zip(raw_results_baseline, raw_results_target):
                    # Pass the original probe prompt to the comparator
                    probe = next((p for p in self.execution_plan.probes if p.id == baseline_res.probe_id), None)
                    original_prompt = probe.prompt if probe else "N/A"
                    
                    comparison_tasks.append(
                        self.comparator.compare_quality_with_llm(
                            baseline_res, target_res, original_probe_prompt=original_prompt
                        )
                    )
                
                quality_evaluations = await asyncio.gather(*comparison_tasks)
                progress.update(compare_task, completed=total_probes)

                # Assemble final evaluation results
                for i in range(total_probes):
                    baseline_res = raw_results_baseline[i]
                    target_res = raw_results_target[i]
                    quality_eval = quality_evaluations[i]
                    
                    eval_res = EvaluationResult(
                        probe_id=baseline_res.probe_id,
                        target_platform=target_res.platform_name,
                        baseline_platform=baseline_res.platform_name,
                        evaluation_success=quality_eval is not None,
                        latency_diff_ms=target_res.latency_ms - baseline_res.latency_ms if baseline_res.latency_ms and target_res.latency_ms else None,
                        cost_diff=target_res.cost_estimate - baseline_res.cost_estimate if baseline_res.cost_estimate and target_res.cost_estimate else None,
                        tokens_diff=target_res.tokens_used - baseline_res.tokens_used if baseline_res.tokens_used and target_res.tokens_used else None,
                        quality_evaluation=quality_eval,
                        timestamp=datetime.now()
                    )
                    evaluation_results.append(eval_res)
            
        end_time = time.monotonic()
        
        # Calculate summary stats
        target_success_count = sum(1 for r in raw_results_target if r.success)
        baseline_success_count = sum(1 for r in raw_results_baseline if r.success)
        eval_success_count = sum(1 for r in evaluation_results if r.evaluation_success)

        report = TrustDiffReport(
            execution_plan=self.execution_plan,
            raw_results_target=raw_results_target,
            raw_results_baseline=raw_results_baseline,
            evaluation_results=evaluation_results,
            execution_timestamp=datetime.now(),
            total_runtime_seconds=end_time - start_time,
            success_rate_target=target_success_count / total_probes if total_probes > 0 else 0,
            success_rate_baseline=baseline_success_count / total_probes if total_probes > 0 else 0,
            evaluation_success_rate=eval_success_count / total_probes if total_probes > 0 else 0
        )
        
        return report

# Alias for backward compatibility if needed, though direct use of TrustDiffEngine is preferred.
Engine = TrustDiffEngine 