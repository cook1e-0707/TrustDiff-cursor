#!/usr/bin/env python3
"""
Quick test script to verify TrustDiff fixes.
Run this script to test the fixes for API calling issues.
"""

import asyncio
import sys
from pathlib import Path

# Add the trustdiff module to the path
sys.path.insert(0, str(Path(__file__).parent))

from trustdiff.main import load_config
from trustdiff.debug_utils import validate_configuration


async def main():
    print("🔧 Testing TrustDiff Fixes...")
    print("=" * 50)
    
    try:
        # Load configuration
        config_path = "configs/default_config.yaml"
        print(f"📂 Loading configuration: {config_path}")
        config = load_config(config_path)
        print("✅ Configuration loaded successfully")
        
        # Run validation
        print("\n🔍 Running configuration validation...")
        await validate_configuration(config)
        
        print("\n✅ Fix verification completed!")
        print("\nNext steps:")
        print("1. Set your environment variables:")
        print("   export OPENAI_API_KEY='your-key'")
        print("   export PLATFORM_A_KEY='your-key'")
        print("   export GEMINI_API_KEY='your-key'")
        print("2. Run: trustdiff debug --config configs/default_config.yaml")
        print("3. Run: trustdiff run --config configs/default_config.yaml")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease check:")
        print("1. Configuration file exists: configs/default_config.yaml")
        print("2. Environment variables are set")
        print("3. Dependencies are installed: pip install -r requirements.txt")


if __name__ == "__main__":
    asyncio.run(main()) 