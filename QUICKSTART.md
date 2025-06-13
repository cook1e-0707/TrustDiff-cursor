# TrustDiff H-CAF Quickstart Guide

This guide will help you get started with running cognitive assessments using the TrustDiff H-CAF framework.

## 1. Installation

First, ensure you have Python 3.10+ and `pip` installed.

```bash
# 1. Clone the repository (if you haven't already)
# git clone <your-repo-url>
# cd TrustDiff-cursor

# 2. Install the package and its dependencies in editable mode.
# This command links the `trustdiff` command to this source code.
pip install -e .
```

## 2. Configuration

All configuration is handled by `configs/default_config.yaml`.

### Set API Keys

The program loads API keys from a `.env` file in your project's root directory.

1.  **Create a file named `.env`** in the `TrustDiff-cursor` directory.
2.  Add your API keys to this file. The names must match what's in `configs/default_config.yaml` (`api_key_env`).

```
# .env file example
OPENAI_API_KEY="sk-..."
GEMINI_API_KEY="ai..."
TARGET_API_KEY="your-other-key"
```

### Configure Platforms

Open `configs/default_config.yaml` to set up the platforms you want to test.

-   **`target`**: The model you want to evaluate.
-   **`baseline`**: The reference model (e.g., an official OpenAI model).
-   **`judge`**: The model that performs the H-CAF quality comparison. It should be a powerful and reliable model (e.g., `gpt-4`).

## 3. Running an Evaluation

The old `init` command has been removed as it is no longer necessary. You can run an evaluation directly.

```bash
# To run a full evaluation using the default configuration:
trustdiff --config configs/default_config.yaml
```

You can also override parts of the configuration using command-line arguments:

```bash
# Use a different set of test probes
trustdiff --config configs/default_config.yaml --probes ./probes/custom/

# Save results to a different directory
trustdiff --config configs/default_config.yaml --output ./my-results/
```

### Viewing Help

To see all available commands and options, use the `--help` flag:

```bash
trustdiff --help
```

## 4. Understanding the Output

After a successful run, you will find the following in your output directory (e.g., `./output`):

-   **`trustdiff_evaluation_...json`**: The main report in JSON format, containing all raw results and H-CAF evaluations.
-   **`trustdiff_evaluation_...yaml`**: The same report in a more human-readable YAML format.
-   **`trustdiff_hcaf_...db`**: A SQLite database containing all results for advanced querying and analysis.
-   **`/raw_responses/`**: A directory containing the raw JSON output from every single API call, useful for debugging.
-   **`/hcaf_reports/`**: A directory with detailed JSON reports focused specifically on the H-CAF cognitive analysis.

The terminal will print a summary of the H-CAF cognitive assessment, highlighting any significant performance degradation detected. 