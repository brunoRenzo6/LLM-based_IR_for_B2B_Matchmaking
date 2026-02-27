"""
snippet_summarizer.py - LLM-based snippet summarization and markdown formatting.
"""

import re
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI

from io_utils import read_json, load_chunk_document
from translator import translate_to_english
import config

parser = StrOutputParser()


# ── Chunk display helpers ─────────────────────────────────────────────────────

def get_chunk_text(u_chunk_id: str) -> str:
    doc_id = u_chunk_id.split("_")[0]
    chunk_key = "_".join(u_chunk_id.split("_")[1:])
    chunks = load_chunk_document(config.SYSTEM, config.DB_ITEM, config.TABLE_ITEM, doc_id)
    return chunks[chunk_key][: config.MAX_CHUNK_CHARS]


def get_url_from_doc_id(doc_id: str) -> str:
    chunks = load_chunk_document(config.SYSTEM, config.DB_ITEM, config.TABLE_ITEM, doc_id)
    return chunks["0"].split("=")[-1]


def build_top_chunks_table(ranking: dict, pid: str, rec: str) -> str:
    """Build a ranked chunk table string for the summarization prompt."""
    chunks_struct = ranking[pid][rec]["chunks"]

    rows = ["cos_sim    doc_id          chunk_text"]
    for ckey in chunks_struct:
        meta = chunks_struct[ckey]
        u_chunk_id = meta["u_chunk_id"]
        doc_id = u_chunk_id.split("_")[0]
        chunk_text = get_chunk_text(u_chunk_id)
        rows.append(f"{meta['cos_sim']:<10.4f} {doc_id:<15} {chunk_text}")

    return "\n".join(rows)


# ── LLM summarization ─────────────────────────────────────────────────────────

def summarize_snippet(model: AzureChatOpenAI, chunks_table: str) -> str:
    """Run the 4-turn LLM summarization chain and return the final summary."""
    prompts = read_json("prompts/prompts_snippet_summary.json")
    messages = []

    messages.append(HumanMessage(content=prompts["m1"].format(chunks_rank=chunks_table)))
    r1 = parser.invoke(model.invoke(messages))

    messages += [SystemMessage(content=r1), SystemMessage(content=prompts["m2"])]
    r2 = parser.invoke(model.invoke(messages))

    messages += [SystemMessage(content=r2), SystemMessage(content=prompts["m3"])]
    r3 = parser.invoke(model.invoke(messages))

    messages += [SystemMessage(content=r3), SystemMessage(content=prompts["m4"])]
    r4 = parser.invoke(model.invoke(messages))

    return r4


def set_snippet_summary(
    model: AzureChatOpenAI,
    ranking: dict,
    pid: str,
    rec: str,
) -> None:
    """
    Compute and store the snippet summary in-place.

    If config.TRANSLATE_TO_ENGLISH is True, the summary is translated to
    English before being stored (translation happens after summarization but
    before markdown / URL injection, so doc-id tokens are still intact).
    """
    chunks_table = build_top_chunks_table(ranking, pid, rec)
    summary = summarize_snippet(model, chunks_table)
    summary = translate_to_english(model, summary)
    ranking[pid][rec]["snippet_summ"] = summary


# ── Markdown / URL injection ──────────────────────────────────────────────────

_CITATION_PATTERNS = [
    r"\[(\d+)\]",
    r"\[(\d+),\s*",
    r",\s*(\d+),\s*",
    r",\s*(\d+)\]",
]


def _replace_doc_ids_with_links(text: str, doc_ids: List[str]) -> str:
    for i, doc_id in enumerate(doc_ids, start=1):
        try:
            url = get_url_from_doc_id(doc_id)
            text = text.replace(doc_id, f"[{i}]({url})", 1)
        except Exception as e:
            print(f"Link replacement error for {doc_id}: {e}")
    return text


def convert_to_markdown(rec_data: dict) -> str:
    """Replace raw doc ids in snippet_summ with markdown links."""
    text = rec_data["snippet_summ"]
    doc_ids: List[str] = []
    for pattern in _CITATION_PATTERNS:
        doc_ids.extend(re.findall(pattern, text))
    return _replace_doc_ids_with_links(text, doc_ids)


def prepend_url(rec_data: dict) -> str:
    """Prepend the root document URL to the markdown snippet."""
    url = get_url_from_doc_id(rec_data["root_doc_id"])
    return f"{url} {rec_data['snippet_summ_mkd']}"
