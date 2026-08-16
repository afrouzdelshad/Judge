import anthropic, os
from dotenv import load_dotenv
load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
with client.messages.stream(
    model="claude-fable-5",
    max_tokens=64000,
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "Say hi in one word."}],
) as stream:
    message = stream.get_final_message()
print("stop_reason:", message.stop_reason)
for b in message.content:
    print(b.type, repr((getattr(b, "text", None) or getattr(b, "thinking", ""))[:120]))
