# tools/llm.py
import os
from openai import OpenAI

_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def llm_json(messages, model="gpt-4o-mini", temperature=0.2, response_format={"type":"json_object"}):
    """
    Ask ChatGPT for a STRICT JSON response. Raise if not JSON.
    """
    resp = _client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format=response_format,
        messages=messages,
    )
    content = resp.choices[0].message.content
    return content  # JSON string (parse at the caller)