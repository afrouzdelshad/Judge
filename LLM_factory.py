import contextlib
import io
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import anthropic
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

load_dotenv()

with open("config.json", "r", encoding="utf-8") as f:
    MODEL_PROVIDERS = {name: info["provider"] for name, info in json.load(f)["llms"].items()}


ANTHROPIC_MAX_TOKENS = 64000
_CACHE_CONTROL = {"type": "ephemeral", "ttl": "1h"}
_BATCH_POLL_SECONDS = 20
_CACHE_PROPAGATION_SECONDS = 2


def resolve_provider(model_name):
    provider = MODEL_PROVIDERS.get(model_name)
    if provider is not None:
        return provider
    m = model_name.lower()
    if "gpt" in m:
        return "openai"
    if "deepseek" in m:
        return "deepseek"
    if "glm" in m:
        return "glm"
    if "claude" in m:
        return "anthropic"
    if "gemini" in m:
        return "gemini"
    raise ValueError(f"Unknown provider for model: {model_name}")


def _anthropic_cached_prompt(system_prompt, user_prompt, use_cache=True):
    """Wrap the prompt in content blocks with a cache breakpoint at the very end.

    Caching matches the whole prefix (tools -> system -> messages) up to and
    including the marked block, so one breakpoint on the last user block covers
    the entire prompt. Only valid because every repeat is byte-identical --
    pass use_cache=False for growing/one-off prompts (e.g. a multi-turn
    transcript) where that never holds, so we don't pay the cache-write
    premium for a write that will never be read back.
    """
    system = [{"type": "text", "text": system_prompt}]
    text_block = {"type": "text", "text": user_prompt}
    if use_cache:
        text_block["cache_control"] = _CACHE_CONTROL
    content = [text_block]
    return system, content


def chat_with(model_name, system_prompt, user_prompt, use_cache=True):
    provider = resolve_provider(model_name)

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
        system, content = _anthropic_cached_prompt(system_prompt, user_prompt, use_cache)
        with client.messages.stream(
            model=model_name,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": content}],
        ) as stream:
            message = stream.get_final_message()
        _log_cache(model_name, "single", message.usage)
        return "".join(block.text for block in message.content if block.type == "text")

    if provider == "gemini":
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        return response.text

    raise ValueError(f"Unsupported provider '{provider}' for model: {model_name}")


def _log_cache(model_name, tag, usage):
    """One line per call so a silent cache miss is visible instead of just expensive."""
    write = getattr(usage, "cache_creation_input_tokens", None)
    read = getattr(usage, "cache_read_input_tokens", None)
    if write is None and read is None:
        return
    status = "hit" if read else ("miss" if write else "none")
    print(
        f"[cache] {model_name} {tag} {status}: write={write} read={read} "
        f"fresh_input={getattr(usage, 'input_tokens', None)}",
        file=sys.stderr,
    )


def _anthropic_batch(model_name, system_prompt, user_prompt, n, poll_seconds):
    """Submit n identical requests as one Message Batch. Returns a list of length n.

    Failed / expired / cancelled entries come back as None rather than raising,
    so one bad request does not throw away the whole batch.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    system, content = _anthropic_cached_prompt(system_prompt, user_prompt)
    params = {
        "model": model_name,
        "max_tokens": ANTHROPIC_MAX_TOKENS,
        "system": system,
        "messages": [{"role": "user", "content": content}],
    }
    batch = client.messages.batches.create(
        requests=[{"custom_id": f"rep-{i}", "params": params} for i in range(n)]
    )
    print(f"[batch] {model_name} submitted {batch.id} ({n} requests)", file=sys.stderr)

    while True:
        status = client.messages.batches.retrieve(batch.id)
        if status.processing_status == "ended":
            break
        time.sleep(poll_seconds)

    # Results stream back in arbitrary order -- custom_id is the only safe key.
    by_id = {}
    for entry in client.messages.batches.results(batch.id):
        if entry.result.type == "succeeded":
            message = entry.result.message
            _log_cache(model_name, entry.custom_id, message.usage)
            if message.stop_reason == "max_tokens":
                print(
                    f"[warn] {model_name} {entry.custom_id} truncated at max_tokens",
                    file=sys.stderr,
                )
            by_id[entry.custom_id] = "".join(
                block.text for block in message.content if block.type == "text"
            )
        else:
            print(
                f"[warn] {model_name} {entry.custom_id} failed: {entry.result.type}",
                file=sys.stderr,
            )
            by_id[entry.custom_id] = None

    return [by_id.get(f"rep-{i}") for i in range(n)]


def _threaded_repeats(model_name, system_prompt, user_prompt, n, max_workers):
    """Fallback for providers without a batch endpoint: concurrent sync calls.

    No 50% discount, but the providers' automatic prefix caching still applies.
    """
    def one(_):
        try:
            return chat_with(model_name, system_prompt, user_prompt)
        except Exception:
            traceback.print_exc()
            return None

    with ThreadPoolExecutor(max_workers=min(max_workers, n)) as pool:
        return list(pool.map(one, range(n)))


def chat_with_batch(model_name, system_prompt, user_prompt, n=10, max_workers=5,
                    poll_seconds=_BATCH_POLL_SECONDS):
    """Run the SAME prompt n times and return a list of n replies (None = failed).

    Repeats are independent samples: cache reuse is a function of the prompt
    prefix only and carries nothing from one reply into the next.

    Pass ONE prebuilt prompt string -- build it outside your loop and reuse the
    same variable. Any per-repeat variation (timestamps, "trial i", shuffled
    order) makes every request a cache miss and costs full price.

    Anthropic goes through the Message Batches API (50% off, stacks with the
    cache discount); other providers fall back to concurrent sync calls.
    Either way call 1 runs synchronously first to write the cache -- fired in
    parallel the repeats would all miss before the first write lands.

    Even sequenced this way, Anthropic's cache write has a server-side
    propagation delay: a request fired immediately after the write returns
    still misses ~40% of the time (see anthropic-sdk-python#1451). We sleep
    a couple seconds before firing the repeats to dodge that race.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    replies = [chat_with(model_name, system_prompt, user_prompt)]
    if n == 1:
        return replies

    time.sleep(_CACHE_PROPAGATION_SECONDS)

    provider = resolve_provider(model_name)
    if provider == "anthropic":
        rest = _anthropic_batch(model_name, system_prompt, user_prompt, n - 1, poll_seconds)
    else:
        rest = _threaded_repeats(model_name, system_prompt, user_prompt, n - 1, max_workers)

    return replies + rest


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


def code_augmented_chat_with(model_name, system_prompt, user_prompt, max_iters=3):
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
    for i in range(max_iters):
        if i == max_iters - 1:
            transcript += (
                "\n\n[System: This is your final iteration. Do not write any more code. "
                "You must give your final answer now, including the required ```json block.]"
            )

        reply = chat_with(model_name, system_prompt, transcript, use_cache=False)
        transcript += f"\n\n[Assistant]\n{reply}"

        code = _extract_python_code(reply)
        if code is None:
            return reply, transcript

        output = _execute_code(code)
        transcript += f"\n\n[System: execution output]\n{output}"

    return reply, transcript


if __name__ == "__main__":
    print(chat_with("claude-opus-4-8", "You are a helpful assistant.", "Hello!"))

    # Build the prompt ONCE, outside the call, then reuse the same string.
    # SYSTEM = build_system_prompt(task)
    # USER = build_user_prompt(series_csv)
    # replies = chat_with_batch("claude-opus-4-8", SYSTEM, USER, n=10)