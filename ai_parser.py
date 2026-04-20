import os
import json
import httpx
from data.data_parser import datas, datalinks

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN_PATH = os.path.join(BASE_DIR, 'TOKEN.txt')

def get_token():
    try:
        with open(TOKEN_PATH, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading token: {e}")
        return ""

def get_ai_response(system_prompt, user_prompt, model="ep-20260407125207-prsm5"):
    """
    Call Volcengine OpenAI-compatible API to get AI summary.
    Returns: (bool success, str content_or_error)
    """
    token = get_token()
    if not token:
        return False, "Failed to get API Token. Please check the TOKEN.txt configuration."
    
    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model, 
        "thinking": {"type": "enabled"},
        "reasoning":{"effort": "low"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
    
    try:
        print(f"Sending AI request with system prompt: {system_prompt[:60]}... and user prompt: {user_prompt[:60]}...")
        # Increase timeout because LLM responses can take a while
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage", {})
            prompt_tokens     = usage.get("prompt_tokens", "?")
            completion_tokens = usage.get("completion_tokens", "?")
            total_tokens      = usage.get("total_tokens", "?")
            print(f"[Token usage] prompt={prompt_tokens}  completion={completion_tokens}  total={total_tokens}")
            return True, data.get("choices", [{}])[0].get("message", {}).get("content", "Failed to extract valid content.")
    except httpx.HTTPStatusError as e:
        # Handle 4xx/5xx Errors, like invalid token (401)
        err_msg = e.response.text
        if e.response.status_code == 401:
            return False, "API authentication failed. Please check if the Token in TOKEN.txt is valid or has expired."
        return False, f"AI request failed (HTTP {e.response.status_code}): {err_msg}"
    except Exception as e:
        return False, f"Failed to get AI summary: {e}"


def get_step1_query_plan(user_query: str):
    """
    Step 1 of the AI search chain.
    Calls the LLM to interpret the user query and return a structured JSON filter plan.
    Returns: (bool success, dict query_plan) or (False, str error_message)
    """
    from pages.ai_search.prompt import get_step1_prompt
    system_prompt, user_prompt = get_step1_prompt(user_query)
    success, content = get_ai_response(system_prompt, user_prompt)
    if not success:
        return False, content

    # Strip markdown code fences the model might add despite instructions
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop opening fence line (```json or ```) and closing fence line (```)
        inner = [ln for ln in lines[1:] if ln.strip() != "```"]
        text = "\n".join(inner)

    try:
        plan = json.loads(text)
        return True, plan
    except json.JSONDecodeError as e:
        return False, (
            f"LLM returned an invalid JSON query plan: {e}\n"
            f"Raw response (first 400 chars): {content[:400]}"
        )


def get_step3_final_answer(user_query: str, results_summary: str):
    """
    Step 3 of the AI search chain.
    Calls the LLM to synthesise a final Markdown answer from the retrieved records.
    Returns: (bool success, str answer) or (False, str error_message)
    """
    from pages.ai_search.prompt import get_step3_prompt
    system_prompt, user_prompt = get_step3_prompt(user_query, results_summary)
    return get_ai_response(system_prompt, user_prompt)
