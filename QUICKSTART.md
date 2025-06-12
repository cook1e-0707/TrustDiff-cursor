# TrustDiff Quick Start Guide

## Installation

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd trustdiff
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package**:
   ```bash
   pip install -e .
   ```

## Setup

1. **Set up environment variables**:
   
   Windows (PowerShell):
   ```powershell
   $env:OPENAI_API_KEY="sk-your-openai-api-key-here"
   $env:PLATFORM_A_KEY="your-platform-a-api-key-here"
   $env:GEMINI_API_KEY="your-gemini-api-key-here"
   ```
   
   Linux/macOS (Bash):
   ```bash
   export OPENAI_API_KEY="sk-your-openai-api-key-here"
   export PLATFORM_A_KEY="your-platform-a-api-key-here"
   export GEMINI_API_KEY="your-gemini-api-key-here"
   ```

2. **Initialize a new project**:
   ```bash
   trustdiff init
   ```

3. **Edit the configuration file**:
   ```bash
   # Edit configs/default_config.yaml with your platform details
   ```

## Basic Usage

### 1. Run Tests

Run all tests:
```bash
trustdiff run --config configs/default_config.yaml
```

Run with filters:
```bash
trustdiff run --config configs/default_config.yaml --probe-filter "reasoning"
```

Dry run (preview without executing):
```bash
trustdiff run --config configs/default_config.yaml --dry-run
```

### 2. Generate Reports

Generate a markdown report:
```bash
trustdiff report outputs/2024-01-01_10-30-00/
```

Generate JSON report:
```bash
trustdiff report outputs/2024-01-01_10-30-00/ --output-format json
```

Save to file:
```bash
trustdiff report outputs/2024-01-01_10-30-00/ --output-file my_report.md
```

### 3. Create Custom Probes

Create a new probe file in the `probes/` directory:

```yaml
# probes/reasoning/my_probe.yaml
probe_id: "my_custom_probe"
probe_type: "reasoning"
description: "My custom test probe"

prompt:
  - role: "user"
    content: "Solve this problem: What is 2+2?"

max_tokens: 100
temperature: 0.1
```

## Project Structure

```
trustdiff/
├── trustdiff/           # Core source code
├── configs/             # Configuration files
├── probes/              # Test probes
│   ├── reasoning/       # Reasoning tests
│   ├── cost/           # Cost tests
│   ├── quality/        # Quality tests
│   └── safety/         # Safety tests
├── outputs/            # Test results (auto-generated)
└── tests/              # Unit tests
```

## Configuration

The main configuration file is `configs/default_config.yaml`:

- **api_keys**: Map environment variables to API keys
- **baseline**: Your ground truth platform (e.g., OpenAI)
- **targets**: Platforms to test against the baseline
- **judge**: LLM to use for quality evaluation
- **run_settings**: Test execution parameters

## Advanced Usage

### Custom Comparator

You can create a custom comparator without LLM judge:

```python
from trustdiff.comparator import SimpleComparator
from trustdiff.engine import Engine

# Use simple comparator for basic testing
engine = Engine(config)
engine.comparator = SimpleComparator()
```

### Programmatic Usage

```python
import asyncio
from trustdiff.engine import Engine
from trustdiff.models import RunConfig

# Load config and run tests programmatically
config = RunConfig.parse_file("configs/default_config.yaml")
engine = Engine(config)

# Run tests
results = asyncio.run(engine.run_test_suite())
```

## Troubleshooting

### Quick Debug Commands

```bash
# 🔍 Validate configuration and test connections
trustdiff debug --config configs/default_config.yaml

# 🧪 Test with a single probe
trustdiff run --config configs/default_config.yaml --probe-filter "reasoning_bucket"

# 👀 Preview without executing
trustdiff run --config configs/default_config.yaml --dry-run
```

### Common Issues

1. **API调用失败但显示Success**: Use `trustdiff debug` to validate configuration
2. **URL拼接错误**: Remove trailing slash from `api_base` in config
3. **第三方平台认证失败**: Check API key format and authentication type
4. **网络连接问题**: Increase `timeout_seconds` and reduce `concurrency`

### Debug Mode

```bash
# Enable detailed debugging
export TRUSTDIFF_DEBUG=1
trustdiff run --config configs/default_config.yaml
```

### 📚 Complete Troubleshooting Guide

For detailed solutions to all common problems, see: `docs/TROUBLESHOOTING.md`

## Next Steps

1. Create custom probes for your specific use cases
2. Configure additional target platforms
3. Set up automated testing with CI/CD
4. Explore advanced reporting features

For more detailed documentation, see the README.md file. 