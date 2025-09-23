from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime

# Creating cache json file to store todays created bugs 
# This will be checked when a new bug is created to avoid duplicates
CACHE = Path("outputs") / "log_analyzer" / "created_bugs.json"
CACHE.parent.mkdir(parents=True, exist_ok=True)

# This method is to create a signature as dated string
def _today_key(signature: str) -> str:
    day = datetime.utcnow().strftime("%Y-%m-%d")
    return f"{day}|{signature}"

# To check the bug is alsready created today
def seen_today(signature: str) -> bool:
    if not CACHE.exists(): return False
    data = json.loads(CACHE.read_text(encoding="utf-8"))
    return _today_key(signature) in data

# If the bug is already created today, mark it in the cache as already existing
def mark_today(signature: str, issue_key: str) -> None:
    data = {}
    if CACHE.exists():
        data = json.loads(CACHE.read_text(encoding="utf-8"))
    data[_today_key(signature)] = issue_key
    CACHE.write_text(json.dumps(data, indent=2), encoding="utf-8")
