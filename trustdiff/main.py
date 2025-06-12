#!/usr/bin/env python3
"""
TrustDiff Main Entry Point with H-CAF Framework Support.
Command-line interface for running cognitive assessment evaluations.
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Dict, Any

from .engine import TrustDiffEngine
from .storage import TrustDiffStorage
from .models import RunConfig


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    import yaml
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)


def setup_environment():
    """Setup environment variables and API keys"""
    import os
    from dotenv import load_dotenv
    
    # Load .env file if it exists
    load_dotenv()
    
    # Check for required environment variables
    required_keys = [
        'OPENAI_API_KEY',
        'GEMINI_API_KEY',  # Optional, depending on configuration
    ]
    
    missing_keys = []
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        print(f"Warning: Missing environment variables: {', '.join(missing_keys)}")
        print("Some API calls may fail without proper API keys.")


async def run_evaluation(config_path: str, probe_dir: str = None, output_dir: str = None):
    """Run TrustDiff evaluation with H-CAF framework"""
    
    # Setup environment
    setup_environment()
    
    # Load configuration
    config = load_config(config_path)
    
    # Override probe directory if specified
    if probe_dir:
        config['probes']['directories'] = [probe_dir]
    
    # Override output directory if specified
    if output_dir:
        config['output']['directory'] = output_dir
    
    # Create run configuration
    run_config = RunConfig(
        max_concurrency=config.get('run_config', {}).get('max_concurrency', 5),
        timeout_seconds=config.get('run_config', {}).get('timeout_seconds', 60),  # Enhanced timeout
        output_format=config.get('run_config', {}).get('output_format', 'json'),
        save_raw_responses=config.get('run_config', {}).get('save_raw_responses', True),
        use_hcaf_framework=config.get('hcaf_config', {}).get('use_hcaf_framework', True),
        hcaf_confidence_threshold=config.get('hcaf_config', {}).get('hcaf_confidence_threshold', 0.6),
        fallback_to_legacy=config.get('hcaf_config', {}).get('fallback_to_legacy', True)
    )
    
    # Initialize storage
    storage = TrustDiffStorage(
        output_dir=config.get('output', {}).get('directory', 'output')
    )
    
    # Initialize engine
    engine = TrustDiffEngine(config, run_config, storage)
    
    try:
        # Load probes
        print("Loading test probes...")
        await engine.load_probes_from_config()
        
        probe_count = len(engine.execution_plan.probes)
        print(f"Loaded {probe_count} test probes")
        
        if probe_count == 0:
            print("No probes found. Please check your probe directories and file patterns.")
            return
        
        # Display H-CAF framework status
        if run_config.use_hcaf_framework:
            print("🧠 H-CAF Framework ENABLED - Cognitive assessment mode")
            print("   Evaluating: Logical Reasoning, Knowledge Application, Creative Synthesis,")
            print("              Instructional Fidelity, Safety & Metacognition")
        else:
            print("📊 Legacy evaluation mode")
        
        # Run evaluation
        print("\nStarting TrustDiff evaluation...")
        print(f"Concurrency: {run_config.max_concurrency}")
        print(f"Timeout: {run_config.timeout_seconds}s")
        
        report = await engine.run_complete_evaluation()
        
        # Generate and save report
        print("\nGenerating reports...")
        report_file = storage.save_report(report, "trustdiff_evaluation")
        print(f"Report saved to: {report_file}")
        
        # Save to database if enabled
        if config.get('output', {}).get('database_enabled', True):
            db_file = storage.save_to_database(report)
            print(f"Database saved to: {db_file}")
        
        # Display summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        print(f"Total probes: {len(report.evaluation_results)}")
        print(f"Target platform: {report.execution_plan.target_platform.name}")
        print(f"Baseline platform: {report.execution_plan.baseline_platform.name}")
        print(f"Success rates:")
        print(f"  Target: {report.success_rate_target:.1%}")
        print(f"  Baseline: {report.success_rate_baseline:.1%}")
        print(f"  Evaluation: {report.evaluation_success_rate:.1%}")
        
        # H-CAF specific summary
        if run_config.use_hcaf_framework:
            cognitive_summary = report.get_cognitive_performance_summary()
            if cognitive_summary and 'hcaf_evaluation_count' in cognitive_summary:
                print(f"\nH-CAF COGNITIVE ASSESSMENT:")
                print(f"  Evaluations completed: {cognitive_summary['hcaf_evaluation_count']}")
                if 'average_degradation' in cognitive_summary:
                    avg_deg = cognitive_summary['average_degradation']
                    print(f"  Average degradation: {avg_deg:.2f}")
                    if avg_deg > 1.0:
                        print(f"  ⚠️  SIGNIFICANT cognitive capability degradation detected")
                    elif avg_deg > 0.5:
                        print(f"  ⚡ Moderate degradation - monitoring recommended")
                    elif avg_deg < -0.5:
                        print(f"  ✨ Performance improvement detected")
                    else:
                        print(f"  ✅ Minimal cognitive impact")
                
                if 'most_impacted_dimension' in cognitive_summary:
                    print(f"  Most impacted: {cognitive_summary['most_impacted_dimension'].replace('_', ' ')}")
        
        print(f"\nExecution time: {report.total_runtime_seconds:.2f} seconds")
        print("="*60)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="TrustDiff: AI Platform Cognitive Assessment with H-CAF Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python -m trustdiff

  # Specify custom configuration file
  python -m trustdiff --config my_config.yaml

  # Use custom probe directory
  python -m trustdiff --probes ./my_probes/

  # Save results to custom directory
  python -m trustdiff --output ./results/

  # Combine options
  python -m trustdiff --config config.yaml --probes ./probes/ --output ./output/
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='configs/default_config.yaml',
        help='Configuration file path (default: configs/default_config.yaml)'
    )
    
    parser.add_argument(
        '--probes', '-p',
        help='Probe directory path (overrides config)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output directory path (overrides config)'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='TrustDiff H-CAF v1.0'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"Error: Configuration file '{args.config}' not found.")
        print("Please create a configuration file or specify an existing one with --config")
        sys.exit(1)
    
    # Print banner
    print("🔍 TrustDiff - AI Platform Cognitive Assessment")
    print("🧠 H-CAF Framework (Hierarchical Cognitive Assessment Framework)")
    print("-" * 60)
    
    # Run evaluation
    try:
        asyncio.run(run_evaluation(
            config_path=args.config,
            probe_dir=args.probes,
            output_dir=args.output
        ))
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 