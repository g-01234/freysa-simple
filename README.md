# LLM Puzzle Game Testing Framework - Freysa Challenge

A streamlined testing framework for experimenting with the [Freysa AI puzzle game](https://www.freysa.ai/genesis/faq). This tool allows systematic testing of prompt combinations against multiple LLM models in parallel.

## Overview

This framework enables:

- Pairwise testing of system and user prompts
- Parallel execution across multiple LLM models
- Configurable prompt filtering and model selection
- Structured output formatting with token usage statistics
- Tool call tracking and visualization

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment and install dependencies:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

3. Create a `.env` file with your API keys:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### Configuration

The system is configured through `config.yaml`. Key settings include:

- Model targets
- Prompt directories
- Include/exclude filters for prompts
- Number of parallel workers
- Model-specific parameters

Example configuration:

```yaml
models:
  targets: ["openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet-latest"]

directories:
  prompts: "prompts"

workers: 3

prompts:
  system:
    target_include: ["sys_freysa_act_1.yaml", "sys_freysa_act_2.json"]
  user:
    target_include:
      ["user_freysa_act_1_winner.yaml", "user_freysa_act_2_winner.json"]
```

## Usage

Run the testing framework:

```bash
uv run simple.py
```

### Expected Output

The system will:

1. Load and combine prompts based on configuration
2. Execute prompts against all configured models in parallel
3. Display formatted results including:
   - Prompt ID and model name
   - Token usage statistics
   - Model responses
   - Tool calls and their arguments

Example output:

```
Loaded 4 prompts
1. sys_freysa_act_1+user_freysa_act_2_winner
2. sys_freysa_act_2+user_freysa_act_2_winner
...

>> Prompt: sys_freysa_act_1+user_freysa_act_2_winner
>> Model: openai/gpt-4o-mini
>> Tokens: in=2788 out=67 total=2855
>> Tools:
* approveTransfer
    {
      "explanation": "..."
    }
...
```

## Project Structure

```
.
├── .env                    # API keys
├── config.yaml            # Configuration file
├── simple.py             # Main testing script
└── prompts/              # Prompt directory
    ├── system/          # System prompts
    └── user/           # User prompts
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT

---

For more information about the Freysa puzzle game, visit [freysa.ai](https://www.freysa.ai/genesis/faq).
