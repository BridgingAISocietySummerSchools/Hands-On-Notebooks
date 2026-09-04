"""
A tiny client for talking to a real large language model.

Used by the optional cells in ``05_agentic_ai.ipynb``. Everything in that
notebook also works *without* a key — these cells then simply skip.

The API key is read from, in this order:

1. the environment variable ``OPENROUTER_API_KEY``
2. a ``.env`` file in the repository root (**gitignored** — never commit it)
3. Google Colab secrets (the 🔑 icon in the left sidebar)

Get a key at https://openrouter.ai/keys and copy ``.env.example`` to ``.env``.
"""

import json
import os
import threading

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-opus-5"

# Running total, so the notebook can show what these calls actually cost.
# The benchmark notebook sends several calls at once, so the update is locked.
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0}
_USAGE_LOCK = threading.Lock()

NO_KEY_MESSAGE = (
    "⏭️  Skipped — no API key found.\n"
    "    This is fine: every other cell in the notebook works without one.\n"
    "    To enable these cells:\n"
    "      • locally — copy .env.example to .env and paste your key into it\n"
    "      • in Colab — add OPENROUTER_API_KEY under 🔑 Secrets in the sidebar\n"
    "    Keys are free to create at https://openrouter.ai/keys"
)


def load_dotenv(path=".env"):
    """Read simple ``KEY=value`` lines from a .env file into ``os.environ``.

    Existing environment variables win, so an explicit export always overrides
    the file. Deliberately minimal — no quoting rules, no interpolation.
    """
    for candidate in (path, os.path.join("..", path)):
        if not os.path.exists(candidate):
            continue
        with open(candidate) as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        return candidate
    return None


def _colab_secret(name):
    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None


def get_api_key():
    """Return the API key, or None if none is configured anywhere."""
    load_dotenv()
    return os.environ.get("OPENROUTER_API_KEY") or _colab_secret("OPENROUTER_API_KEY")


def get_model():
    load_dotenv()
    return os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)


def llm_available():
    """True if a real model can be called right now."""
    return bool(get_api_key())


def describe_setup():
    """Print whether a model is reachable, and which one."""
    if not llm_available():
        print(NO_KEY_MESSAGE)
        return False
    print(f"🔌 Connected. Model: {get_model()}")
    print(f"   Key loaded from a .env file or the environment — never from this notebook.")
    return True


def llm(prompt, system=None, model=None, max_tokens=800, temperature=0.0,
        thinking=False):
    """Send one prompt to the model and return its reply as a string.

    ``thinking=False`` asks the provider not to spend the token budget on an
    internal reasoning trace — without it, a short ``max_tokens`` can be used up
    before the model writes any visible answer at all.
    """
    import requests

    key = get_api_key()
    if not key:
        raise RuntimeError("No OPENROUTER_API_KEY configured.")

    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})

    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model or get_model(), "messages": messages,
              "max_tokens": max_tokens, "temperature": temperature,
              "reasoning": {"enabled": bool(thinking)}},
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()

    usage = payload.get("usage", {})
    with _USAGE_LOCK:
        USAGE["calls"] += 1
        USAGE["prompt_tokens"] += usage.get("prompt_tokens", 0)
        USAGE["completion_tokens"] += usage.get("completion_tokens", 0)
        USAGE["cost"] += usage.get("cost", 0.0) or 0.0

    choice = payload["choices"][0]
    message = choice.get("message", {})
    # `content` can be None when the whole budget went into a reasoning trace.
    text = (message.get("content") or message.get("reasoning") or "").strip()
    if not text:
        return f"(the model returned no text; finish_reason={choice.get('finish_reason')})"
    if choice.get("finish_reason") == "length":
        text += "  […truncated: raise max_tokens]"
    return text


def ask_llm(prompt, **kwargs):
    """Like ``llm()``, but returns a friendly notice instead of raising."""
    if not llm_available():
        return NO_KEY_MESSAGE
    try:
        return llm(prompt, **kwargs)
    except Exception as error:                      # network hiccup, quota, typo in model id
        return f"⚠️  The API call failed: {type(error).__name__}: {error}"


def ask_llm_json(prompt, **kwargs):
    """Ask for JSON and parse it. Returns None if the reply was not valid JSON."""
    reply = ask_llm(prompt, **kwargs)
    match = reply[reply.find("{"):reply.rfind("}") + 1] if "{" in reply else ""
    try:
        return json.loads(match)
    except (json.JSONDecodeError, ValueError):
        return None


def print_usage():
    """Show what the real-model calls in this notebook have cost so far."""
    if not USAGE["calls"]:
        print("No real model calls were made in this session.")
        return
    print(f"📊 {USAGE['calls']} API call(s) so far")
    print(f"   Input tokens:  {USAGE['prompt_tokens']:,}")
    print(f"   Output tokens: {USAGE['completion_tokens']:,}")
    print(f"   Total cost:    ${USAGE['cost']:.4f}")
