"""Core package exports for agents.

This package re-exports the minimal APIs agents need so agent code can use
`from src.core import chat, pick_requirement` without deep imports.

Keep this file small: its purpose is purely convenience for teaching and
to keep example agent files short and readable.
"""

from .llm_client import chat #imports chat method from the llm_client.py file.
from .utils import pick_requirement, parse_json_safely, to_rows, write_csv, write_json # similarly other methods in utils.py file

# MAkes sure the above method is available, something like namespace
# See the file testcase_agent.py > line 7 > on how we use the methods
__all__ = [
    "chat",
    "pick_requirement",
    "parse_json_safely",
    "to_rows",
    "write_csv",
    "write_json",
]