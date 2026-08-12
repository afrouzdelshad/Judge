import contextlib
import io
import json
import os
import re
import traceback

import anthropic
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

load_dotenv()

with open("config.json", "r", encoding="utf-8") as f:
    MODEL_PROVIDERS = {name: info["provider"] for name, info in json.load(f)["llms"].items()}


def chat_with(model_name, system_prompt, user_prompt):
    provider = MODEL_PROVIDERS.get(model_name)
    if provider is None:
        m = model_name.lower()
        if "gpt" in m:
            provider = "openai"
        elif "deepseek" in m:
            provider = "deepseek"
        elif "glm" in m:
            provider = "glm"
        elif "claude" in m:
            provider = "anthropic"
        elif "gemini" in m:
            provider = "gemini"
        else:
            raise ValueError(f"Unknown provider for model: {model_name}")

    if provider == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    if provider == "deepseek":
        client = OpenAI(base_url="https://api.deepseek.com", api_key=os.getenv("DEEPSEEK_API_KEY"))
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    if provider == "glm":
        client = OpenAI(base_url="https://open.bigmodel.cn/api/paas/v4", api_key=os.getenv("GLM_API_KEY"))
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    if provider == "anthropic":
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        with client.messages.stream(
            model=model_name,
            max_tokens=64000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            message = stream.get_final_message()
        return message.content[0].text

    if provider == "gemini":
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        return response.text

    raise ValueError(f"Unsupported provider '{provider}' for model: {model_name}")


def _extract_python_code(text):
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1) if match else None


def _execute_code(code):
    """Run model-generated code locally (unsandboxed) and capture stdout."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": __builtins__})
    except Exception:
        buf.write("\n[EXCEPTION]\n" + traceback.format_exc())
    return buf.getvalue() or "(no output)"


def code_augmented_chat_with(model_name, system_prompt, user_prompt, max_iters=6):
    """Like chat_with, but lets the model execute Python code against local data.

    Repeats up to `max_iters` times: send the prompt, and if the reply
    contains a fenced ```python code block, exec it locally (unsandboxed)
    and feed the captured stdout back as the next message. Stops as soon as
    a reply has no code block, treating it as the final answer.

    Note: code is exec'd with no sandboxing. Only use this with trusted API
    providers on a machine you control.

    Returns (final_reply, transcript).
    """
    transcript = user_prompt
    reply = None
    for _ in range(max_iters):
        reply = chat_with(model_name, system_prompt, transcript)
        transcript += f"\n\n[Assistant]\n{reply}"

        code = _extract_python_code(reply)
        if code is None:
            return reply, transcript

        output = _execute_code(code)
        transcript += f"\n\n[System: execution output]\n{output}"

    return reply, transcript


if __name__ == "__main__":
    print(chat_with("claude-opus-4-8", "You are a helpful assistant.", "Hello!"))
