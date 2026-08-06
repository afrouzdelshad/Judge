import json
import os

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


if __name__ == "__main__":
    print(chat_with("claude-opus-4-8", "You are a helpful assistant.", "Hello!"))
