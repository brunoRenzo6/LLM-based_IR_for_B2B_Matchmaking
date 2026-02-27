"""
streamlit_app.py - Streamlit front-end for the ranking system.

The UI builds a DataStore.from_query(user_input) and then runs the same
pipeline stage functions (run_stage_l0 … run_stage_l3) used in batch mode.
No duplicate pipeline logic lives here.

Run with:
    streamlit run streamlit_app.py
"""

import random
from typing import List

import streamlit as st

import config
from data_store import DataStore
from pipeline import run_stage_l0, run_stage_l1, run_stage_l2, run_stage_l3
from translator import translate_to_english
from io_utils import read_json
from llm_eval import build_llm_client
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser


# ── Cached resource loading ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading data warehouse …")
def load_data_store() -> DataStore:
    return DataStore()


@st.cache_resource(show_spinner="Building LLM client …")
def load_llm_client():
    return build_llm_client()


# ── Project anonymization ─────────────────────────────────────────────────────

def anonymize_project(model, original_project: str) -> str:
    """
    Use the LLM to generate a fictional project request inspired by the original,
    removing any identifying information while preserving style and domain context.
    """
    prompts = read_json("prompts/prompts_anonymize_project.json")
    parser = StrOutputParser()
    messages = [
        SystemMessage(content=prompts["system"]),
        HumanMessage(content=prompts["m1"].format(original_project=original_project)),
    ]
    return parser.invoke(model.invoke(messages))


# ── Single-query pipeline ─────────────────────────────────────────────────────

def get_recommendations(query: str, base_ds: DataStore, model) -> List[str]:
    """
    Run the full pipeline for a single free-text query.

    Stubs only the query-specific project attributes onto a shell object that
    shares all heavy item-side data (embeddings, chunks, id lookup) directly
    from the cached base_ds — no disk I/O or Polars joins on each search.
    """
    # Build a shell DataStore, bypassing __init__ entirely so _load_items()
    # and _build_aligned_chunks() are never called.  Only the 5 lightweight
    # project-stub assignments run before we hand off to the pipeline.
    ds = DataStore.__new__(DataStore)
    ds._stub_project(query)
    # Share item-side data by reference — zero extra I/O or computation.
    ds.doc_emb = base_ds.doc_emb
    ds.u_chunk_id_l = base_ds.u_chunk_id_l
    ds.id_lookup_new = base_ds.id_lookup_new
    ds.df_chunks_aligned = base_ds.df_chunks_aligned

    with st.status("🔎 Running vector search …", expanded=True) as status:
        ranking = run_stage_l0(ds)
        status.update(label="📊 Grading results with LLM …")
        ranking = run_stage_l1(ranking, ds)
        status.update(label="🔁 Re-ranking …")
        ranking = run_stage_l2(ranking)
        status.update(label="✍️ Summarising snippets …")
        ranking = run_stage_l3(ranking)
        status.update(label="Done!", state="complete", expanded=False)

    snippets = []
    for pid in ranking:
        for rec, data in ranking[pid].items():
            mkd = data.get("snippet_summ_mkd", "")
            if mkd:
                snippets.append(mkd)
    return snippets


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def write_search_snippets(recommendations: List[str]) -> None:
    if recommendations:
        st.write("### Search Results:")
        for snippet in recommendations:
            st.write(snippet)


def main():
    st.set_page_config(page_title="B2B Matchmaking")

    base_ds = load_data_store()
    model = load_llm_client()

    if "project_desc" not in st.session_state:
        st.session_state.project_desc = ""
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = []

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.title("Menu")

    sample_clicked = st.sidebar.button("🎲 Sample a project request")
    sidebar_placeholder = st.sidebar.empty()

    if sample_clicked:
        projects_l = list(base_ds.df_projects["project_desc"])
        sampled = random.choice(projects_l)

        with sidebar_placeholder.container():
            with st.spinner("✍️ Generating anonymized project …"):
                anonymized = anonymize_project(model, sampled)
                st.session_state.project_desc = translate_to_english(model, anonymized)

    if st.session_state.project_desc:
        sidebar_placeholder.success(st.session_state.project_desc)

    # ── Main area ─────────────────────────────────────────────────────────────
    st.markdown(
        """
        <style>
        div[data-testid="column"] + div[data-testid="column"] {
            padding-left: 0rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("B2B Matchmaking")

    query = st.text_area(
        "",
        value=st.session_state.project_desc,
        placeholder="e.g. I want to hire a web developer with experience in Flutter …",
    )

    col_btn, col_status = st.columns([0.8, 4], vertical_alignment="top", gap="small")

    with col_btn:
        search_clicked = st.button("🔍 Search")

    if search_clicked:
        if not query.strip():
            st.warning("Please enter a query before searching.")
        else:
            with col_status:
                st.session_state.recommendations = get_recommendations(query, base_ds, model)
            write_search_snippets(st.session_state.recommendations)
    else:
        write_search_snippets(st.session_state.recommendations)


if __name__ == "__main__":
    main()