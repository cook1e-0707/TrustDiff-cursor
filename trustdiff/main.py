"""
TrustDiff CLI main entry point using Typer.
"""

import os
import asyncio
from pathlib import Path
from typing import Optional
import typer
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from .models import RunConfig
from .engine import Engine
from .reporter import Reporter
from .debug_utils import validate_configuration
from .gemini_debug import test_gemini_api, test_gemini_models

app = typer.Typer(
    name="trustdiff",
    help="TrustDiff - Testing framework for LLM API platform reliability",
    add_completion=False
)
console = Console()


def load_config(config_path: str) -> RunConfig:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise typer.BadParameter(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # Resolve environment variables in API keys
    if 'api_keys' in config_data:
        for key, value in config_data['api_keys'].items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                env_value = os.getenv(env_var)
                if not env_value:
                    console.print(f"[red]Warning: Environment variable {env_var} not set[/red]")
                config_data['api_keys'][key] = env_value or ""
    
    return RunConfig(**config_data)


@app.command()
def init(
    output_dir: str = typer.Option("./", help="Directory to initialize project in")
):
    """Initialize a new TrustDiff project with example configuration."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create directory structure
    dirs = ["configs", "probes/reasoning", "probes/cost", "outputs", "tests"]
    for dir_name in dirs:
        (output_path / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Create example config
    config_path = output_path / "configs" / "default_config.yaml"
    if not config_path.exists():
        example_config = {
            "api_keys": {
                "openai": "${OPENAI_API_KEY}",
                "platform_a_key": "${PLATFORM_A_KEY}",
                "judge_llm_key": "${OPENAI_API_KEY}"
            },
            "baseline": {
                "name": "OpenAI_Official",
                "api_base": "https://api.openai.com/v1",
                "api_key_env": "openai",
                "model": "gpt-3.5-turbo"
            },
            "targets": [
                {
                    "name": "Platform_A", 
                    "api_base": "https://api.platform-a.com/v1",
                    "api_key_env": "platform_a_key",
                    "model": "gpt-3.5-turbo"
                }
            ],
            "judge": {
                "name": "Judge_LLM",
                "api_base": "https://api.openai.com/v1", 
                "api_key_env": "judge_llm_key",
                "model": "gpt-4"
            },
            "run_settings": {
                "probe_dir": "./probes",
                "output_dir": "./outputs",
                "concurrency": 10,
                "timeout_seconds": 60,
                "use_hcaf_framework": True,
                "cognitive_vectors_focus": None
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(example_config, f, default_flow_style=False, allow_unicode=True)
    
    console.print(f"[green]✓[/green] Initialized TrustDiff project in {output_path}")
    console.print(f"[blue]Next steps:[/blue]")
    console.print(f"  1. Edit {config_path} with your API keys")
    console.print(f"  2. Add probe files to {output_path / 'probes'}")
    console.print(f"  3. Run: trustdiff run --config {config_path}")


@app.command()
def run(
    config: str = typer.Option("./configs/default_config.yaml", help="Path to configuration file"),
    probe_filter: Optional[str] = typer.Option(None, help="Filter probes by pattern"),
    dry_run: bool = typer.Option(False, help="Show what would be run without executing")
):
    """Run TrustDiff tests against configured platforms."""
    try:
        # Load configuration
        run_config = load_config(config)
        
        if dry_run:
            console.print("[yellow]Dry run mode - showing configuration:[/yellow]")
            console.print(f"Baseline: {run_config.baseline.name}")
            console.print(f"Targets: {[t.name for t in run_config.targets]}")
            console.print(f"Probe directory: {run_config.probe_dir}")
            console.print(f"Output directory: {run_config.output_dir}")
            return
        
        # Run the engine
        console.print("[blue]Starting TrustDiff test run...[/blue]")
        
        async def run_tests():
            engine = Engine(run_config)
            return await engine.run_test_suite(probe_filter=probe_filter)
        
        # Run async test suite
        results = asyncio.run(run_tests())
        
        if results:
            console.print(f"[green]✓[/green] Test run completed successfully")
            console.print(f"[blue]Results saved to:[/blue] {results.output_dir}")
        else:
            console.print("[red]✗[/red] Test run failed")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def report(
    results_dir: str = typer.Argument(..., help="Path to results directory"),
    output_format: str = typer.Option("markdown", help="Output format (markdown, json, html)"),
    output_file: Optional[str] = typer.Option(None, help="Output file path")
):
    """Generate a report from test results."""
    try:
        results_path = Path(results_dir)
        if not results_path.exists():
            raise typer.BadParameter(f"Results directory not found: {results_path}")
        
        console.print(f"[blue]Generating report from {results_path}...[/blue]")
        
        reporter = Reporter(results_path)
        report_content = reporter.generate_report(format=output_format)
        
        if output_file:
            output_path = Path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            console.print(f"[green]✓[/green] Report saved to {output_path}")
        else:
            console.print(report_content)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def debug(
    config: str = typer.Option("./configs/default_config.yaml", help="Path to configuration file")
):
    """Debug and validate TrustDiff configuration."""
    try:
        run_config = load_config(config)
        
        async def run_debug():
            await validate_configuration(run_config)
        
        asyncio.run(run_debug())
        
    except Exception as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise typer.Exit(1)


@app.command() 
def test_gemini(
    model: str = typer.Option("gemini-1.5-pro", help="Gemini model to test"),
    test_models: bool = typer.Option(False, help="Test all available Gemini models")
):
    """Test Gemini API functionality and model availability."""
    try:
        import os
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            console.print("[red]❌ GEMINI_API_KEY environment variable not set[/red]")
            console.print("[yellow]Please run: export GEMINI_API_KEY='your-gemini-api-key'[/yellow]")
            raise typer.Exit(1)
        
        async def run_tests():
            if test_models:
                await test_gemini_models(api_key)
            await test_gemini_api(api_key, model)
        
        asyncio.run(run_tests())
        
    except Exception as e:
        console.print(f"[red]Gemini test failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def version():
    """Show TrustDiff version."""
    from . import __version__
    console.print(f"TrustDiff version {__version__}")


if __name__ == "__main__":
    app() 