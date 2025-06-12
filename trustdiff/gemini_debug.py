"""
Gemini API specific debugging utilities.
"""

import asyncio
import httpx
from rich.console import Console
from rich.json import JSON

console = Console()


async def test_gemini_api(api_key: str, model: str = "gemini-1.5-pro") -> None:
    """Test Gemini API with different configurations."""
    console.print(f"\n[bold blue]🧪 Testing Gemini API[/bold blue]")
    console.print(f"Model: {model}")
    console.print("=" * 50)
    
    test_prompt = "Respond with exactly: Hello from Gemini!"
    
    # Test different API configurations
    configs = [
        {
            "name": "Standard Configuration",
            "body": {
                "contents": [{"parts": [{"text": test_prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 100,
                    "candidateCount": 1
                }
            }
        },
        {
            "name": "Enhanced Configuration", 
            "body": {
                "contents": [{"parts": [{"text": test_prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 2000,
                    "candidateCount": 1,
                    "responseMimeType": "text/plain"
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
            }
        }
    ]
    
    for config in configs:
        console.print(f"\n[cyan]Testing: {config['name']}[/cyan]")
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                
                response = await client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=config["body"]
                )
                
                console.print(f"Status Code: {response.status_code}")
                
                if response.status_code == 200:
                    response_data = response.json()
                    console.print("[green]✅ Success![/green]")
                    console.print("Response structure:")
                    console.print(JSON.from_data(response_data, indent=2))
                    
                    # Try to extract text
                    if 'candidates' in response_data and response_data['candidates']:
                        candidate = response_data['candidates'][0]
                        
                        text_found = False
                        
                        # Try different extraction methods
                        if 'content' in candidate and 'parts' in candidate['content']:
                            if candidate['content']['parts']:
                                text = candidate['content']['parts'][0].get('text', '')
                                if text:
                                    console.print(f"[green]Extracted text (method 1): {text}[/green]")
                                    text_found = True
                        
                        if not text_found:
                            console.print("[yellow]Could not extract text from response[/yellow]")
                            console.print(f"Candidate structure: {candidate}")
                    
                else:
                    console.print(f"[red]❌ Failed: {response.status_code}[/red]")
                    console.print(f"Error: {response.text}")
                    
        except Exception as e:
            console.print(f"[red]❌ Exception: {e}[/red]")


async def test_gemini_models(api_key: str) -> None:
    """Test different Gemini models."""
    models = [
        "gemini-1.5-pro",
        "gemini-1.5-flash", 
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro-latest"
    ]
    
    console.print(f"\n[bold blue]🔍 Testing Available Gemini Models[/bold blue]")
    console.print("=" * 50)
    
    for model in models:
        console.print(f"\n[cyan]Testing model: {model}[/cyan]")
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                
                test_body = {
                    "contents": [{"parts": [{"text": "Hi"}]}],
                    "generationConfig": {"maxOutputTokens": 10}
                }
                
                response = await client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=test_body
                )
                
                if response.status_code == 200:
                    console.print(f"[green]✅ {model} is available[/green]")
                elif response.status_code == 404:
                    console.print(f"[red]❌ {model} not found[/red]")
                else:
                    console.print(f"[yellow]⚠️ {model} returned {response.status_code}[/yellow]")
                    
        except Exception as e:
            console.print(f"[red]❌ {model} failed: {e}[/red]")


if __name__ == "__main__":
    import os
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("[red]Please set GEMINI_API_KEY environment variable[/red]")
        exit(1)
    
    async def main():
        await test_gemini_models(api_key)
        await test_gemini_api(api_key, "gemini-1.5-pro")
    
    asyncio.run(main()) 