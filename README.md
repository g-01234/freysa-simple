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

## Key Components

### Prompt Loading

```59:153:simple.py
def load_prompts(user_prompts_dir: str, system_prompts_dir: str) -> list[Prompt]:
    """Load and combine prompts from user and system directories.

    This function reads prompt files from separate user and system directories,
    applies filtering based on config settings, and creates all possible combinations
    of system and user prompts.

    Args:
        user_prompts_dir: Directory path containing user prompt files
        system_prompts_dir: Directory path containing system prompt files

    Returns:
        List of Prompt objects, each representing a unique combination of
        system and user prompts with associated configuration

    Raises:
        FileNotFoundError: If either directory doesn't exist
        yaml.YAMLError: If YAML files are malformed
        json.JSONDecodeError: If JSON files are malformed
    """

    # Read include/exclude settings from config
    user_include = config.get("prompts", {}).get("user", {}).get("target_include", [])
    user_exclude = config.get("prompts", {}).get("user", {}).get("target_exclude", [])
    system_include = (
        config.get("prompts", {}).get("system", {}).get("target_include", [])
    )
    system_exclude = (
        config.get("prompts", {}).get("system", {}).get("target_exclude", [])
    )
    # Get list of user prompt files
    user_files = [
        f
        for f in os.listdir(user_prompts_dir)
        if f.endswith((".yaml", ".yml", ".json"))
    ]
    print(user_files)
    # Filter user prompt files based on include/exclude settings
    if user_include:
        user_files = [f for f in user_files if f in user_include]
    if user_exclude:
        user_files = [f for f in user_files if f not in user_exclude]
    print(user_files)

    # Get list of system prompt files
    system_files = [
        f
        for f in os.listdir(system_prompts_dir)
        if f.endswith((".yaml", ".yml", ".json"))
    ]

    # Filter system prompt files based on include/exclude settings
    if system_include:
        system_files = [f for f in system_files if f in system_include]
    if system_exclude:
        system_files = [f for f in system_files if f not in system_exclude]

    prompts = []
    prompt_settings = config.get(
        "prompt_settings", {"temperature": None, "top_p": None}
    )

    # Create pairwise combinations of system prompts and user prompts
    for user_file in user_files:
        user_prompts = _read_prompt_from_file(os.path.join(user_prompts_dir, user_file))
        user_messages = user_prompts["messages"]
        for system_file in system_files:
            system_prompts = _read_prompt_from_file(
                os.path.join(system_prompts_dir, system_file)
            )
            system_messages = system_prompts["messages"]
            tools = system_prompts.get("tools")

            # Combine system and user messages
            messages = system_messages + user_messages

            # Create a unique identifier for the prompt combination - remove file extension
            prompt_id = f"{system_file.split('.')[0]}+{user_file.split('.')[0]}"

            prompts.append(
                Prompt(
                    prompt_id=prompt_id,
                    messages=messages,
                    tools=tools,
                    temperature=prompt_settings["temperature"],
                    top_p=prompt_settings["top_p"],


    print(f"Loaded \033[36m{len(prompts)}\033[0m prompts")
    for i, prompt in enumerate(prompts):
        print(f"{i+1}. {prompt.prompt_id}")

```

### Parallel Execution

```156:210:simple.py
def send_prompt_par(prompt: Prompt) -> list[Generation]:
    """Send a prompt to multiple models in parallel.

    Uses ThreadPoolExecutor to concurrently send the same prompt to all configured
    models, handling failures gracefully.

    Args:
        prompt: Prompt object containing messages and configuration

    Returns:
        List of Generation objects containing responses from each model.
        Failed generations include error information in the response.

    Note:
        The number of parallel workers is controlled by the WORKERS config setting.
        Each model attempt includes retry logic defined in _send_to_model.
    """
    max_retries = 5
    backoff_factor = 2
    delay = 1
    results = []

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = []
        for model in config["models"]["targets"]:
            futures.append(
                executor.submit(
                    _send_to_model,
                    prompt,
                    model,
                    max_retries,
                    backoff_factor,
                    delay,
                )
            )

        for future, model in zip(as_completed(futures), config["models"]["targets"]):
            try:
                response = future.result()
                results.append(
                    Generation(
                        prompt_id=prompt.prompt_id, model=model, response=response
                    )
                )
            except Exception as e:
                print(f"Error sending prompt to model {model}: {e}")
                results.append(
                    Generation(
                        prompt_id=prompt.prompt_id,
                        model=model,
                        response={"error": str(e)},


```

### Response Formatting

```242:289:simple.py
def _format_generation(generation: Generation) -> str:
    """Formats a generation result with ANSI color codes for terminal display.

    Creates a formatted string containing the prompt ID, model name, token usage statistics,
    model response content, and any tool calls made by the model.

    Args:
        generation (Generation): The generation result to format.

    Returns:
        str: A multi-line formatted string with ANSI color codes for terminal display.
            Includes sections for:
            - Prompt ID and model name
            - Token usage statistics (if available)
            - Model's response content (if available)
            - Tool calls with function names and arguments (if any were made)
    """
    output = []
    output.append(f"\033[33m>>\033[0m \033[90mPrompt:\033[0m {generation.prompt_id}")
    output.append(f"\033[33m>>\033[0m \033[90mModel:\033[0m {generation.model}")

    # Add token usage statistics
    usage = generation.response.usage
    if usage:
        output.append(
            f"\033[33m>>\033[0m \033[90mTokens:\033[0m in={usage.prompt_tokens} out={usage.completion_tokens} total={usage.total_tokens}"
        )

    # Extract and format the model's response
    response_content = generation.response.choices[0].message.get("content")
    if response_content:
        output.append("\033[33m>>\033[0m \033[90mResponse:\033[0m")
        output.append(f"{response_content.strip()}")

    # Extract and format tool calls if they exist
    tool_calls = generation.response.choices[0].message.get("tool_calls", [])
    output.append("\033[33m>>\033[0m \033[90mTools:\033[0m")
    if tool_calls:
        for tool in tool_calls:
            output.append(f"\033[32m*\033[0m {tool.function.name}")
            # Pretty print the arguments
            args = json.dumps(json.loads(tool.function.arguments), indent=2)
            for line in args.splitlines():
                output.append(f"     \033[34m{line}\033[0m")
    else:
        output.append("\033[31mNone\033[0m")

    output.append("\033[90m"
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Your chosen license]

---

For more information about the Freysa puzzle game, visit [freysa.ai](https://www.freysa.ai/genesis/faq).
