import os
import yaml
from litellm import completion, ModelResponse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dataclasses import dataclass
from typing import Any, Optional

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

PROMPTS_DIR = config["directories"]["prompts"]
PROMPTS_USER_DIR = os.path.join(PROMPTS_DIR, "user")
PROMPTS_SYSTEM_DIR = os.path.join(PROMPTS_DIR, "system")
WORKERS = config.get("workers", 3)


@dataclass
class Generation:
    """Represents a single generation result from an LLM model.

    Attributes:
        prompt_id (str): Unique identifier for the prompt combination.
        model (str): Name of the LLM model used for generation.
        response (ModelResponse): Response object from the LLM containing
            completion, token usage, and tool calls if applicable.
    """

    prompt_id: str
    model: str
    response: ModelResponse


@dataclass
class Prompt:
    """Configuration for a prompt to be sent to LLM models.

    Attributes:
        prompt_id (str): Unique identifier for the prompt combination.
        messages (list[dict[str, Any]]): List of message dictionaries containing
            the conversation context.
        tools (Optional[list[dict[str, Any]]], optional): Function calling tools
            available to the model. Defaults to None.
        tool_choice (Union[str, dict[str, Any], None], optional): Tool selection mode.
            Can be "none", "auto", or a specific function. Defaults to None.
        temperature (Optional[float], optional): Sampling temperature. Defaults to None.
        top_p (Optional[float], optional): Nucleus sampling parameter. Defaults to None.
    """

    prompt_id: str
    messages: list[dict[str, Any]]
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: str | dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None


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
                )
            )

    print(f"Loaded \033[36m{len(prompts)}\033[0m prompts")
    for i, prompt in enumerate(prompts):
        print(f"{i+1}. {prompt.prompt_id}")

    return prompts


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
                    )
                )

    return results


def _read_prompt_from_file(
    file_path: str,
) -> dict[str, list[dict[str, Any]]]:
    """Load prompt data from a YAML or JSON file.

    Args:
        file_path: Path to the prompt file (.yaml, .yml, or .json)

    Returns:
        Dictionary containing prompt data with at minimum a 'messages' key
        containing the conversation messages. May also include 'tools' and
        other configuration.

    Raises:
        ValueError: If file extension is not .yaml, .yml, or .json
        yaml.YAMLError: If YAML file is malformed
        json.JSONDecodeError: If JSON file is malformed
        FileNotFoundError: If file doesn't exist
    """
    if file_path.endswith((".yaml", ".yml")):
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    elif file_path.endswith(".json"):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported file format. Please provide a YAML or JSON file.")


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

    output.append("\033[90m" + "-" * 50 + "\033[0m")
    return "\n".join(output)


def _send_to_model(
    prompt: Prompt,
    model: str,
    max_retries: int,
    backoff_factor: int,
    delay: int,
) -> ModelResponse:
    """Send a prompt to a specific model with retry logic.

    Attempts to send the prompt multiple times in case of failures, with
    configurable retry behavior.

    Args:
        prompt: Prompt object containing messages and configuration
        model: Name/identifier of the target model
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for exponential backoff between retries
        delay: Initial delay in seconds before first retry

    Returns:
        ModelResponse object containing the completion, token usage, and
        any tool calls made by the model

    Raises:
        Exception: Propagates any unhandled exceptions from the model API
            after max retries are exhausted

    Note:
        Currently implements basic retry without actual backoff/delay.
        Future versions will implement proper exponential backoff.
    """
    for _ in range(max_retries):
        response = completion(
            model=model,
            messages=prompt.messages,
            tools=prompt.tools,
            tool_choice="auto" if prompt.tools else None,
            temperature=prompt.temperature,
            top_p=prompt.top_p,
            **{
                "custom_llm_provider": "anthropic" if "claude" in model else None,
            },
        )

        return response


def main():
    all_results = []

    prompts = load_prompts(PROMPTS_USER_DIR, PROMPTS_SYSTEM_DIR)
    print(f"\nLoaded \033[36m{len(prompts)}\033[0m prompts")

    for prompt in prompts:
        result = send_prompt_par(prompt)
        all_results.extend(result)
    for generation in all_results:
        print(_format_generation(generation))

    print(f"Generated \033[36m{len(all_results)}\033[0m responses")


if __name__ == "__main__":
    main()
