# This file is to create testcase where requirements posted under data/requirements folder using LLM
# and testcases are created as csv and json files under outputs/testcase_generated folder


# Use below code to run this file step by step:
# 1. create virtual environment - 
#  python -m venv .\venv
# 2. activate virtual environment
# .\venv\Scripts\Activate.ps1
# 3 If any error in activating virtual environment, run this command in PowerShell: 
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# 4. install dependencies one by one
# pip install langchain langchain-openai langchain-ollama python-dotenv langgraph rich loguru
# -here rich and loguru dependancies are to see logs in a better organised way
# 5. TO see the versions of installed packages
# pip show  langchain langchain-openai langchain-ollama python-dotenv langgraph rich loguru
# 6. set OPENAI_API_KEY in .env variable ( This may not be required for Ollama, but just in case you want to test OpenAI also)
#   $env:OPENAI_API_KEY="your_api_key_here" 
#   6.a Check the Open AI API is set correctly
# echo $env:OPENAI_API_KEY  > would show the saved API key 
# 7 Now run the ollama.
# open powershell seperately from run window > – Windows +R > powershell
#   7.a Type ollama > run > shows the commands
#   7.b Type ollama run mistral:7b  (this will start the ollama server)
#   7.c Type a sample prompt like "What is machine learning?" to check whether it is working fine.
# 8. Now run the file (make sure ollama is running before the runnning this command)
# python -m src.agents.testcase_agent --input data/requirements/login.txt

# The project contains a demo API services for Testrail, Jira and Slack
# GO to JiraTestrailSlackAPI-ToTest folder in this project > double click start-all.bat > will start all the API services, to see
#   Testrail UI, open browser and type http://localhost:4001/ui & API can be seen from http://localhost:4001/docs#
#   Jira UI, open browser and type http://localhost:4002/ui & API can be seen from http://localhost:4002/docs#
#   Slack UI, open browser and type http://localhost:4003/ui & API can be seen from http://localhost:4003/docs#

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict

# Because of the __init__ and all, can import the methods directly
from src.core import chat, pick_requirement, parse_json_safely, to_rows, write_csv
import argparse
from langchain.prompts import PromptTemplate
from typing import Optional # For Logging

# importing the methods for testrail integration created in src/integrations/testrail.py
from src.integrations.testrail import map_case_to_testrail_payload, create_case, list_cases, add_result, get_stats
import re  # regular expression library

# The functions imported from `src.core` are small, dependency-free helpers
# used to keep this agent focused on orchestration (easy for students to read
# and extend): `chat` (LLM call), `pick_requirement` (choose input file),
# `parse_json_safely` (robust JSON parsing), `to_rows` and `write_csv`.

# Paths (easy-to-change constants)
ROOT = Path(__file__).resolve().parents[2] # setting the root folder as 2 levels above from the current folder
REQ_DIR = ROOT / "data" / "requirements"  # directory with .txt requirement files
OUT_DIR = ROOT / "outputs" / "testcase_generated"  # where outputs are written
OUT_DIR.mkdir(parents=True, exist_ok=True) # This is to create the above folders if not exist
OUT_CSV = OUT_DIR / "test_cases.csv"  # CSV output path

LAST_RAW_JSON = OUT_DIR / "last_raw.json"  # file where raw LLM text is saved

PROMPTS_DIR = ROOT / "src" / "core" / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR / "testcase_system.txt").read_text(encoding="utf-8")
USER_TEMPLATE_STR = (PROMPTS_DIR / "testcase_user.txt").read_text(encoding="utf-8")
USER_TEMPLATE = PromptTemplate.from_template(USER_TEMPLATE_STR)


# `USER_TEMPLATE` wraps the requirement text so the model sees a clear input
# block; we keep it simple for students to inspect and modify.

Message = Dict[str, str] # for payload as string of string dictionary
"""Type alias for message dicts sent to the `chat` helper.

Each message is a dict with `role` and `content` strings, matching the
minimal interface used by provider-agnostic chat helpers in the exercises.
"""
# Chat method without logging
# def main(argv: Optional[list] = None) -> None:
#     """Run the testcase agent end-to-end.

#     Flow:
#     1. Pick a requirement file (CLI arg or first file in `data/requirements`).
#     2. Read the requirement text.
#     3. Build a system + user message pair and call `chat(messages)`.
#     4. Save raw model output to `outputs/last_raw.json` and parse it as JSON.
#     5. Convert parsed testcases to CSV rows and write `outputs/test_cases.csv`.

#     Error handling and teaching hooks:
#     - We save the raw model text to `LAST_RAW_JSON` so students can inspect
#       model failures or formatting issues.
#     - If parsing fails, we perform a single "nudge" retry that reminds the
#       model to return pure JSON. If it still fails we raise a helpful error
#       pointing to the raw file.
#     - The agent is deliberately thin: it focuses on orchestration. Students
#       can extend it later to add retries, rate-limiting, human-in-the-loop
#       review pages, or direct integrations with Jira/TestRail.
#     """
#     # print("Hi")
#     # print(sys.argv[0])
#     # print(sys.argv[1])
#     # print(sys.argv[2])
#     # print("Hi2")

#     # print(f"System prompt is  {SYSTEM_PROMPT}")
#     # print("\ln")
#     # print(f"user prompt is  {USER_TEMPLATE_STR}")
          
#     req_path = pick_requirement(sys.argv[2] if len(sys.argv) > 1 else None, REQ_DIR)
#     #req_path = pick_requirement(args.input if args.input else None, REQ_DIR)
#     #
#     #print(req_path)
#     requirement_text = req_path.read_text(encoding="utf-8").strip()
    
#     #print(f"requirement text is {requirement_text}")

#     messages: List[Message] = [
#         {"role": "system",
#          "content": SYSTEM_PROMPT
#          },
#         {
#             "role": "user",
#             "content": USER_TEMPLATE.format(requirement_text=requirement_text),
#         },
#     ]

#     # Call the LLM via the provider-agnostic `chat` function. The returned
#     # `raw` is the assistant's text; for Day-1 we expect the model to return
#     # a pure JSON array (see SYSTEM_PROMPT) so downstream parsing is simple.
#     raw = chat(messages)

#     try:
#         cases = parse_json_safely(raw, LAST_RAW_JSON)
#     except Exception as e:
#         # gentle retry nudge — a pragmatic teaching technique: show how a
#         # small reminder can correct common model format mistakes.
#         nudge = (
#             raw + "\n\nREMINDER: Return a pure JSON array only, matching the schema."
#         )
#         try:
#             cases = parse_json_safely(nudge, LAST_RAW_JSON)
#         except Exception:
#             # Surface a clear runtime error with a pointer to the saved raw
#             # output so students can debug model responses during the session.
#             raise RuntimeError(
#                 f"Could not parse model output as JSON. See {LAST_RAW_JSON}.\nError: {e}"
#             )

#     rows = to_rows(cases)
#     write_csv(rows, OUT_CSV)

#     print(f"✅ Wrote {len(rows)} test cases to: {OUT_CSV.relative_to(ROOT)}")
#     print(f"ℹ️  Raw model output saved at: {LAST_RAW_JSON.relative_to(ROOT)}")

# this method is change the case to small, remove special characters and trim the spaces
def _norm(title: str | None) -> str:
    """
    Normalize a title for stable dedupe.
    - case-insensitive
    - trims
    - removes non-alphanumeric (keeps [a-z0-9] only)
    - collapses whitespace
    """
    s = (title or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Chat method with logging enabled
def main(argv: Optional[list] = None) -> None:
    """Run the testcase agent end-to-end.

    Flow:
    1. Pick a requirement file (CLI arg or first file in `data/requirements`).
    2. Read the requirement text.
    3. Build a system + user message pair and call `chat(messages)`.
    4. Save raw model output to `outputs/last_raw.json` and parse it as JSON.
    5. Convert parsed testcases to CSV rows and write `outputs/test_cases.csv`.

    Error handling and teaching hooks:
    - We save the raw model text to `LAST_RAW_JSON` so students can inspect
      model failures or formatting issues.
    - If parsing fails, we perform a single "nudge" retry that reminds the
      model to return pure JSON. If it still fails we raise a helpful error
      pointing to the raw file.
    - The agent is deliberately thin: it focuses on orchestration. Students
      can extend it later to add retries, rate-limiting, human-in-the-loop
      review pages, or direct integrations with Jira/TestRail.
    """

    # Parse CLI: accept `--input PATH` for clarity in teaching demos
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to a requirement .txt file")
    args = parser.parse_args(argv)

    # Configure logging for the process (simple default; agents may override)
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger(__name__)

    #req_path = pick_requirement(args.input if args.input else None, REQ_DIR)
    req_path = pick_requirement(sys.argv[2] if len(sys.argv) > 1 else None, REQ_DIR)
    requirement_text = req_path.read_text(encoding="utf-8").strip()

    messages: List[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(requirement_text=requirement_text),
        },
    ]

    # Call the LLM via the provider-agnostic `chat` function. The returned
    # `raw` is the assistant's text; for Day-1 we expect the model to return
    # a pure JSON array (see SYSTEM_PROMPT) so downstream parsing is simple.
    logger.debug("Calling chat: provider payload msgs=%d (sys=1,user=1)", len(messages))
    raw = chat(messages)

    try:
        cases = parse_json_safely(raw, LAST_RAW_JSON)
    except Exception as e:
        # gentle retry nudge — a pragmatic teaching technique: show how a
        # small reminder can correct common model format mistakes.
        logger.exception(
            "Initial parse_json_safely failed; will nudge and retry. Raw saved at %s",
            LAST_RAW_JSON,
        )
        nudge = (
            raw + "\n\nREMINDER: Return a pure JSON array only, matching the schema."
        )
        try:
            cases = parse_json_safely(nudge, LAST_RAW_JSON)
        except Exception:
            # Surface a clear runtime error with a pointer to the saved raw
            # output so students can debug model responses during the session.
            logger.error(
                "Could not parse model output after nudge; see %s", LAST_RAW_JSON
            )
            raise RuntimeError(
                f"Could not parse model output as JSON. See {LAST_RAW_JSON}.\nError: {e}"
            )

    rows = to_rows(cases)
    write_csv(rows, OUT_CSV)

    logger.info("✅ Wrote %d test cases to: %s", len(rows), OUT_CSV.relative_to(ROOT))
    logger.info("ℹ️  Raw model output saved at: %s", LAST_RAW_JSON.relative_to(ROOT))


  #Testrail API is not working for some reason, so commented the code of pushing testcases to Testrail via API calls
  # Below code is working one, can be enabled once the testrail API issue is fixed
    # # Code to push the generated testcases to TestRail via api calls
    # # Includes simple duplicate checking with Title.
    # logger.info("ℹ️  Starting TestRail push step")

    # # Map once → collect payloads (so we dedupe on the exact titles we will POST)
    # payloads: list[dict] = []
    # for idx, c in enumerate(cases, start=1):
    #     try:
    #         p = map_case_to_testrail_payload(c)
    #         payloads.append(p)
    #     except Exception as e:
    #         logger.warning("Skipping case %s (mapping error): %s", c.get("id") or idx, e)

    # # Picking the Titles from LLM created test cases & removing the special characters, spaces and converting to small case
    # incoming_titles = { _norm(p.get("title")) for p in payloads }

    # # Picking the Titles from Testrail app & removing the special characters, spaces and converting to small case
    # try:
    #     existing = list_cases()  # returns list[dict]
    #     existing_titles = { _norm(case.get("title")) for case in existing }
    # except Exception as e:
    #     logger.warning("Could not fetch existing titles; proceeding without dedupe: %s", e)
    #     existing_titles = set()

    # logger.info("📚 Loaded %d existing titles from TestRail (project-wide)", len(existing_titles))

    # # Checking with AND operation between the LLM generated titles and Testrail existing titles
    # dupes = incoming_titles & existing_titles
    # if dupes:
    #     logger.info(
    #         "🚧 Detected %d duplicate title(s) in this batch; they will be skipped: %s",
    #         len(dupes), sorted(list(dupes))[:5]  # if duplicate exists, then shows first 5 only
    #     )
    # else:
    #     logger.info("✅ No duplicates detected for this batch")

    # created_ids: list[int] = []
    # for p in payloads:
    #     title_norm = _norm(p.get("title"))

    #     # Skip if the testcase Title already exists 
    #     if title_norm in existing_titles:
    #         logger.info("↪️  Skipping existing case: %s", p.get("title"))
    #         continue

    #     try:
    #         res = create_case(p)
    #         cid = res.get("id")
    #         if cid is not None:
    #             created_ids.append(int(cid))
    #             existing_titles.add(title_norm)  # avoid same-batch duplicates
                
    #             # Update the testcase status as Untested = 3
    #             try:
    #                 _ = add_result(int(cid), status_id=3, comment="Seeded by agent on create") # Not needed  any return, so uses "_="
    #             except Exception as e:
    #                 logger.warning("Could not seed result for case %s: %s", cid, e)
    #         else:
    #             logger.warning("Create case response missing 'id': %s", res)
    #     except Exception as e:
    #         logger.error("Create case failed for '%s': %s", p.get("title"), e)

    # logger.info("📌 Created %d TestRail cases: %s", len(created_ids), created_ids)

    # # Shows count of testcases in the mentioned project
    # try:
    #     all_cases = list_cases()
    #     logger.info("🧾 TestRail now has %d cases in project", len(all_cases))
    # except Exception as e:
    #     logger.warning("Could not list TestRail cases: %s", e)

    # logger.info("✅ Test cases pushed to TestRail successfully with id %s", created_ids)
    
    # # Fetch and log project stats
    # try:
    #     stats = get_stats()
    #     total = stats.get("total_cases")
    #     logger.info("📊 Project stats → total_cases: %s", total)
    #     # (Optional) log section breakdowns
    #     for s in stats.get("sections", []):
    #         logger.info("   • %s: %s case(s)", s.get("section_name"), s.get("case_count"))
    # except Exception as e:
    #     logger.warning("Could not fetch project stats: %s", e)


if __name__ == "__main__":
    main()



