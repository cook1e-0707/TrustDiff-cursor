#!/usr/bin/env python3
"""
Simple entry point script for TrustDiff.
This can be used if the package isn't installed via pip.
"""

import sys
from pathlib import Path

# Add the trustdiff module to the Python path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    from trustdiff.main import app
    app() 