"""
llm_eval.py - LLM-based relevance grading of ranked search results.
"""

import copy
import os
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI

import config
from io_utils import read_json, load_chunk_document

parser = StrOutputParser()


# ── LLM client ────────────────────────────────────────────────────────────────

def build_llm_client() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=config.GPT_DEPLOYMENT,
        api_version=config.GPT_API_VERSION,
        temperature=config.GPT_TEMPERATURE,
    )


# ── Chunk text retrieval ──────────────────────────────────────────────────────

def get_chunk_text(u_chunk_id: str) -> str:
    """Fetch and truncate a single chunk by its unique chunk id."""
    doc_id = u_chunk_id.split("_")[0]
    chunk_key = "_".join(u_chunk_id.split("_")[1:])
    chunks = load_chunk_document(config.SYSTEM, config.DB_ITEM, config.TABLE_ITEM, doc_id)
    return chunks[chunk_key][: config.MAX_CHUNK_CHARS]


# ── Search snippet construction ───────────────────────────────────────────────

def build_search_snippet(ranking: dict, project_id: str, rec: str, id_lookup: dict) -> str:
    """
    Assemble a formatted snippet string from the chunks stored under
    ranking[project_id][rec], prepended with the source URL.
    """
    chunks_struct = ranking[project_id][rec]["chunks"]
    root_doc_id = ranking[project_id][rec]["root_doc_id"]
    url = id_lookup[root_doc_id]["url"]

    sep = "\n--------------"
    lines = [f"{url}{sep}"]
    for e, ckey in enumerate(chunks_struct):
        u_chunk_id = chunks_struct[ckey]["u_chunk_id"]
        chunk_text = get_chunk_text(u_chunk_id)
        lines.append(f"chunk_{e}: {chunk_text}{sep}")

    return "\n".join(lines)


# ── LLM grading ───────────────────────────────────────────────────────────────

def grade_relevance(
    model: AzureChatOpenAI,
    prompts: dict,
    project_desc: str,
    seller_desc: str,
) -> Optional[int]:
    """
    Run the multi-turn LLM eval chain and return an integer grade.
    Returns None if the model output cannot be parsed.
    """
    p = copy.deepcopy(prompts)
    messages = []

    p["m1"] = p["m1"].format(
        project_description=project_desc,
        seller_description=seller_desc,
    )
    messages.append(HumanMessage(content=p["m1"]))
    r1 = parser.invoke(model.invoke(messages))

    messages.append(AIMessage(content=r1))
    messages.append(HumanMessage(content=p["m2"]))
    r2 = parser.invoke(model.invoke(messages))

    messages.append(AIMessage(content=r2))
    messages.append(HumanMessage(content=p["m3"]))
    r3 = parser.invoke(model.invoke(messages))

    try:
        raw = r3.split("=")[1].replace("\n", "").replace("```", "")
        grade = eval(raw)  # noqa: S307 – controlled LLM output
        return grade if isinstance(grade, int) else None
    except Exception:
        return None


# ── Batch grading ─────────────────────────────────────────────────────────────

def grade_ranking_entry(
    model: AzureChatOpenAI,
    prompts: dict,
    ranking: dict,
    project_id: str,
    rec: str,
    project_desc: str,
    id_lookup: dict,
    exceptions: list,
) -> None:
    """
    Grade a single (project_id, rec) pair in-place inside `ranking`.
    Idempotent: skips entries that already have a grade.

    Args:
        project_desc: The buyer project description used as the grading reference.
                      In batch mode this comes from ds.get_project_desc(project_id);
                      in interactive mode it is the raw user-supplied query string.
    """
    if "grade" in ranking[project_id][rec]:
        return  # already graded

    try:
        snippet = build_search_snippet(ranking, project_id, rec, id_lookup)
        grade = grade_relevance(model, prompts, project_desc, snippet)

        if grade is not None:
            ranking[project_id][rec]["grade"] = grade
    except Exception as e:
        print(f"Error grading {project_id}/{rec}: {e}")
        exceptions.append((project_id, rec, e))