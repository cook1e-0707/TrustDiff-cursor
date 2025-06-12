"""
Debug utilities for TrustDiff.
"""

import asyncio
import httpx
from rich.console import Console
from rich.table import Table
from .models import RunConfig, PlatformConfig

console = Console()


async def test_platform_connection(platform_config: PlatformConfig, api_key: str) -> dict:
    """Test connection to a platform."""
    test_result = {
        "platform": platform_config.name,
        "api_base": platform_config.api_base,
        "status": "unknown",
        "response_code": None,
        "error": None,
        "latency_ms": None
    }
    
    try:
        import time
        start_time = time.time()
        
        # Simple test request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        test_body = {
            "model": platform_config.model or "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5
        }
        
        api_url = platform_config.api_base.rstrip('/') + '/chat/completions'
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                api_url,
                headers=headers,
                json=test_body
            )
            
            latency = (time.time() - start_time) * 1000
            test_result["latency_ms"] = latency
            test_result["response_code"] = response.status_code
            
            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data:
                    test_result["status"] = "success"
                else:
                    test_result["status"] = "invalid_response"
                    test_result["error"] = "No 'choices' in response"
            else:
                test_result["status"] = "http_error"
                test_result["error"] = f"HTTP {response.status_code}: {response.text[:200]}"
                
    except httpx.TimeoutException:
        test_result["status"] = "timeout"
        test_result["error"] = "Request timeout"
    except httpx.ConnectError as e:
        test_result["status"] = "connection_error"
        test_result["error"] = f"Connection failed: {str(e)}"
    except Exception as e:
        test_result["status"] = "error"
        test_result["error"] = str(e)
    
    return test_result


async def validate_configuration(config: RunConfig) -> None:
    """Validate the entire configuration."""
    console.print("\n[bold blue]🔍 TrustDiff Configuration Validation[/bold blue]\n")
    
    # Check API keys
    console.print("[bold]API Keys Status:[/bold]")
    missing_keys = []
    
    for key_name, key_value in config.api_keys.items():
        if not key_value or key_value.startswith("${"):
            missing_keys.append(key_name)
            console.print(f"  ❌ {key_name}: Not set or invalid")
        else:
            # Mask the key for security
            masked_key = key_value[:8] + "..." + key_value[-4:] if len(key_value) > 12 else "***"
            console.print(f"  ✅ {key_name}: {masked_key}")
    
    if missing_keys:
        console.print(f"\n[red]⚠️  Missing API keys: {', '.join(missing_keys)}[/red]")
        console.print("[yellow]Please set the environment variables or update the config file.[/yellow]\n")
    
    # Test platform connections
    console.print("\n[bold]Platform Connection Tests:[/bold]")
    
    all_platforms = [config.baseline] + config.targets
    if config.judge:
        all_platforms.append(config.judge)
    
    test_tasks = []
    for platform in all_platforms:
        api_key = config.api_keys.get(platform.api_key_env, "")
        if api_key and not api_key.startswith("${"):
            test_tasks.append(test_platform_connection(platform, api_key))
    
    if test_tasks:
        results = await asyncio.gather(*test_tasks, return_exceptions=True)
        
        # Create results table
        table = Table()
        table.add_column("Platform", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("API Base", style="dim")
        table.add_column("Latency", style="yellow")
        table.add_column("Error", style="red")
        
        for result in results:
            if isinstance(result, dict):
                status_color = {
                    "success": "[green]✅ Success[/green]",
                    "http_error": "[red]❌ HTTP Error[/red]",
                    "connection_error": "[red]❌ Connection Error[/red]",
                    "timeout": "[yellow]⏱️ Timeout[/yellow]",
                    "invalid_response": "[yellow]⚠️ Invalid Response[/yellow]",
                    "error": "[red]❌ Error[/red]"
                }.get(result["status"], "[dim]❓ Unknown[/dim]")
                
                latency_str = f"{result['latency_ms']:.1f}ms" if result["latency_ms"] else "N/A"
                error_str = result["error"][:50] + "..." if result["error"] and len(result["error"]) > 50 else (result["error"] or "")
                
                table.add_row(
                    result["platform"],
                    status_color,
                    result["api_base"],
                    latency_str,
                    error_str
                )
        
        console.print(table)
    
    # Configuration summary
    console.print(f"\n[bold]Configuration Summary:[/bold]")
    console.print(f"  📂 Probe directory: {config.probe_dir}")
    console.print(f"  📤 Output directory: {config.output_dir}")
    console.print(f"  🔄 Concurrency: {config.concurrency}")
    console.print(f"  ⏱️ Timeout: {config.timeout_seconds}s")
    console.print(f"  🎯 Target platforms: {len(config.targets)}")
    console.print(f"  👨‍⚖️ Judge configured: {'Yes' if config.judge else 'No'}")


def print_debug_info(raw_result, platform_config):
    """Print detailed debug information for a failed request."""
    console.print(f"\n[bold red]🐛 Debug Info for {platform_config.name}[/bold red]")
    console.print(f"URL: {platform_config.api_base}")
    console.print(f"Model: {platform_config.model}")
    console.print(f"Success: {raw_result.success}")
    console.print(f"Latency: {raw_result.latency_ms}ms")
    
    if raw_result.error_message:
        console.print(f"Error: {raw_result.error_message}")
    
    if raw_result.response_data:
        console.print(f"Response: {str(raw_result.response_data)[:200]}...")
    
    console.print() 