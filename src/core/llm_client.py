# This file is to create a common interface to call the LLM models using LANGCHAIN

from __future__ import annotations # To be checked
import os # to read the env files
from typing import List, Dict # importing list and dict  
import httpx # To contact the outside world using API
from dotenv import load_dotenv # to read the env file

# LangChain chat model wrappers + message types
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.schema import SystemMessage, HumanMessage, AIMessage
import logging

# Visibility / logging flags (class-friendly defaults)
LLM_LOG = os.getenv("LLM_LOG", "1").strip().lower() in ("1", "true", "yes")
LLM_DEBUG = os.getenv("LLM_DEBUG", "0").strip().lower() in ("1", "true", "yes")

# module logger (agents should configure logging.basicConfig in their entrypoints)
logger = logging.getLogger(__name__)

load_dotenv(override=True) # loads the env file, override=True is to priority to env file than OS

PROVIDER = (os.getenv("PROVIDER") or "ollama").strip().lower() # take the provider from env, if not available set as ollama
MODEL = (os.getenv("MODEL") or "mistral:latest").strip()
OLLAMA_HOST = (os.getenv("OLLAMA_HOST") or "http://localhost:11434").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_S") or "180")

LLM_TEMPERATURE = None

Message = Dict[str, str] # To store the {'role','content'}

# This method is to create the prompt with langchain message format
def _to_lc_messages(messages: List[Message]): # lc means langchain
    """
    Convert [{'role','content'}] into LangChain BaseMessages.
    See the file testcase_agent.py >  here the messages set roles as system, user etc 
    > These roles are moved to list in loop with langchain format
    """
    lc_msgs = []
    for m in messages:
        role = (m.get("role") or "").lower()
        content = m.get("content") or ""
        if role == "system":
            lc_msgs.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_msgs.append(AIMessage(content=content))
        else:
            # treat 'user' and anything else as human input
            lc_msgs.append(HumanMessage(content=content))
    return lc_msgs

# This method is to set the LLM and api key if required
def _make_llm():
    """
    Create the LangChain chat model according to PROVIDER/MODEL envs.

    Note: We do NOT pass a `timeout` kwarg here for maximum compatibility
    across LangChain versions/backends (e.g., ChatOllama often has no such arg).
    """
    if PROVIDER == "ollama":
        # LangChain's Ollama wrapper reads OLLAMA_HOST from env.
        os.environ["OLLAMA_HOST"] = OLLAMA_HOST
        return ChatOllama(model=MODEL)
    elif PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing but PROVIDER=openai.")
        # Keep temperature=0 for deterministic teaching runs
        return ChatOpenAI(model=MODEL, temperature=0)
    else:
        raise NotImplementedError("Unsupported PROVIDER. Use 'ollama' or 'openai'.")


def chat(messages: List[Message], timeout: int = TIMEOUT_S) -> str:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list of {'role','content'} dicts.")

    # for logging > progress: start
    if LLM_LOG:
        n_sys = sum(1 for m in messages if (m.get("role") or "").lower() == "system") # For logging > system message
        n_usr = sum(1 for m in messages if (m.get("role") or "").lower() in ("user", "human")) # For logging > user message
        n_ast = sum(1 for m in messages if (m.get("role") or "").lower() in ("assistant", "ai"))
        msg_count = len(messages)

        size_info = "" 
        if LLM_DEBUG:
            lengths = [len(m.get("content") or "") for m in messages] # Logging the length of prompt
            size_info = f" | chars={sum(lengths)} total, per_msg={lengths}" # Logging the no of words / characters in prompt

        logger.info(
            "[LLM] ▶ start provider=%s model=%s msgs=%d (sys=%d, user=%d, asstnt=%d)%s",
            PROVIDER,
            MODEL,
            msg_count,
            n_sys,
            n_usr,
            n_ast,
            size_info,
        )

    # ---- call model
    import time
    t0 = time.perf_counter()
    llm = _make_llm()
    lc_msgs = _to_lc_messages(messages)

    try:
        resp = llm.invoke(lc_msgs)
        out = getattr(resp, "content", "") or ""
        dt = time.perf_counter() - t0
        if LLM_LOG:
            logger.info("[LLM] ✔ done in %.2fs", dt)
        if LLM_DEBUG:
            logger.debug("[LLM] response length=%d", len(out))
        return out
    except Exception as e:
        dt = time.perf_counter() - t0
        # log exception with stacktrace
        logger.exception("[LLM] ✖ error after %.2fs: %s", dt, type(e).__name__)
        raise


# def chat(messages: List[Message], timeout: int = TIMEOUT_S) -> str:
#     """
#     Send OpenAI-style messages and return assistant text (string).
#     Keeps the exact caller contract your agents already use.

#     timeout is not passing for the maximum compatibility
#     `timeout` is currently advisory (not enforced for all backends uniformly).
#     """
#     if not isinstance(messages, list) or not messages:
#         raise ValueError("messages must be a non-empty list of {'role','content'} dicts.")

#     llm = _make_llm() 
#     lc_msgs = _to_lc_messages(messages) # lc means langchain

#     # Call the model; avoid per-call config to keep compatibility wide.
#     resp = llm.invoke(lc_msgs)
#     return getattr(resp, "content", "") or ""















# This method is to call the LLM without LANGCHAIN, deactivated as we use LANGCHAIN
# def chat(messages: List[Message], timeout: int = TIMEOUT_S) -> str:
#     """Send `messages` to the configured LLM provider and return assistant text.

#     This thin, provider-agnostic helper keeps the interface simple for Day-1
#     teaching: callers pass OpenAI-style `messages` and get back the assistant's
#     `content` string. Validation is intentionally minimal to keep code readable.

#     Args:
#         messages: List of message dicts with `role` and `content`.
#         timeout: Request timeout in seconds.

#     Returns:
#         str: Assistant text returned by the selected provider.

#     Raises:
#         ValueError: If `messages` is empty or not a list.
#         RuntimeError: For provider-specific failures (missing keys, empty replies).
#         NotImplementedError: If `PROVIDER` is not supported.
#     """
    
#     # Checks the messages of type list and messages variable is empty or not -(not messages)
#     if not isinstance(messages, list) or not messages: 
#         raise ValueError(
#             "messages must be a non-empty list of {'role','content'} dicts."
#         )

#     if PROVIDER == "ollama":
#         url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
#         payload = {"model": MODEL, "messages": messages, "stream": False}
#         with httpx.Client(timeout=timeout) as client: #####
#             r = client.post(url, json=payload)
#             r.raise_for_status()
#             data = r.json()
#             msg = (data.get("message") or {}).get("content")
#             if not msg:
#                 raise RuntimeError(
#                     "Ollama returned empty content. Check model and host."
#                 )
#             return msg

#     elif PROVIDER == "openai":
#         if not OPENAI_API_KEY:
#             raise RuntimeError("OPENAI_API_KEY is missing but PROVIDER=openai.")
#         url = "https://api.openai.com/v1/chat/completions" # Move to env file
#         headers = {
#             "Authorization": f"Bearer {OPENAI_API_KEY}",
#             "Content-Type": "application/json",
#         }
#         payload = {"model": MODEL, "messages": messages, "temperature": 0}
#         with httpx.Client(timeout=timeout) as client:   #####
#             r = client.post(url, headers=headers, json=payload)
#             r.raise_for_status()
#             data = r.json()
#             choices = data.get("choices") or []
#             if not choices:
#                 raise RuntimeError("OpenAI returned no choices. Check model and key.")
#             return (choices[0].get("message") or {}).get("content") or ""

#     else:
#         raise NotImplementedError("Unsupported PROVIDER. Use 'ollama' or 'openai'.")


