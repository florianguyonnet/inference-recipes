#!/usr/bin/env python3
"""
Test tool calling against the local vLLM server.

Verifies that Qwen3.8-27B-NVFP4 emits valid OpenAI tool calls that Pi Agent
can parse without errors.
"""

import argparse
import json
import sys
from typing import Any

import requests


def run_test(url: str, model: str) -> bool:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [
        {
            "role": "user",
            "content": "What's the weather like in Paris right now?",
        }
    ]

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": {
            "type": "function",
            "function": {"name": "get_weather"},
        },
        "max_tokens": 512,
        "temperature": 0.0,
    }

    print("Sending request with tools ...")
    resp = requests.post(
        f"{url}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    choice = data["choices"][0]
    message = choice["message"]
    finish_reason = choice["finish_reason"]

    print(f"finish_reason: {finish_reason}")
    print(f"message keys: {list(message.keys())}")

    tool_calls = message.get("tool_calls")
    if not tool_calls:
        print("ERROR: no tool_calls in response")
        return False

    print(f"tool_calls: {json.dumps(tool_calls, indent=2)}")

    # Basic OpenAI schema validation
    for tc in tool_calls:
        if tc.get("type") != "function":
            print(f"ERROR: unexpected tool_call type {tc.get('type')}")
            return False
        fn = tc.get("function", {})
        if not fn.get("name"):
            print("ERROR: missing function name")
            return False
        if not isinstance(fn.get("arguments"), str):
            print("ERROR: function.arguments must be a JSON string")
            return False
        try:
            json.loads(fn["arguments"])
        except json.JSONDecodeError as exc:
            print(f"ERROR: invalid JSON arguments: {exc}")
            return False

    print("OK: tool call is valid OpenAI format")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen3.8-27B-NVFP4")
    args = parser.parse_args()

    ok = run_test(args.url, args.model)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
