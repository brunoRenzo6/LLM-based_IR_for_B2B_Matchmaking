"""
pipeline.py - End-to-end orchestration of the ranking pipeline.

Stages
------
L_0  Vector search + result structuring
L_1  LLM relevance grading
L_2  Grade-based re-ranking
L_3  Snippet summarization + markdown formatting

Batch mode (default — processes every project in the dataset):
    python pipeline.py

Subset mode (useful for spot-checks):
    python pipeline.py --pids proj_001 proj_002
"""

import argparse
import concurrent.futures
from typing import List, Optional

from config import TOP_K_0, TOP_K, SNIPPET_TOP_K
from data_store import DataStore
from io_utils import read_json, write_json
from llm_eval import build_llm_client, grade_ranking_entry
from ranking import get_rankings_multi_query, build_struct_sellers
from reranker import rerank_by_grade
from snippet_summarizer import set_snippet_summary, convert_to_markdown, prepend_url


# ── Stage L_0: Vector retrieval ───────────────────────────────────────────────

def run_stage_l0(
    ds: DataStore,
    project_ids: Optional[List[str]] = None,
    out_path: str = "L_0.json",
) -> dict:
    """
    Retrieve and structure top candidates for every project.

    Each project's text chunks are used together as the multi-query set,
    mirroring the strategy used during notebook development. Results are keyed
    by project_id; the project description is read on the fly from the DataStore
    when needed (e.g. for grading) rather than stored in the ranking struct.

    Args:
        ds:          Loaded DataStore.
        project_ids: Optional explicit list of project ids to process.
                     Defaults to every project in ds.df_projects.
        out_path:    Where to persist the stage output.

    Returns:
        ranking dict  {project_id: {"r0": {...}, "r1": {...}, ...}}
    """
    if project_ids is None:
        project_ids = list(ds.df_projects["id"])

    ranking: dict = {}

    for i, pid in enumerate(project_ids):
        queries = ds.get_project_chunks(pid)
        print(f"[L_0] {i + 1}/{len(project_ids)}  project={pid}  n_queries={len(queries)}")

        df_chunks, _df_docs, df_root_docs = get_rankings_multi_query(queries, ds)
        struct = build_struct_sellers(
            df_chunks, df_root_docs,
            top_k=TOP_K_0,
            snippet_top_k=SNIPPET_TOP_K,
        )
        ranking[pid] = struct

    write_json(ranking, out_path)
    print(f"[L_0] Saved → {out_path}")
    return ranking


# ── Stage L_1: LLM grading ────────────────────────────────────────────────────

def run_stage_l1(
    ranking: dict,
    ds: DataStore,
    out_path: str = "L_1.json",
) -> dict:
    """
    Grade every retrieved result with the LLM eval chain.

    All projects present in the ranking dict are graded — no filtering needed
    because L_0 already controls which projects enter the ranking. Grading is
    parallelised across all (project, record) pairs. Results are checkpointed
    every 500 records; re-running is safe because already-graded records are
    skipped (idempotent).

    Args:
        ranking:  Output of run_stage_l0.
        ds:       Loaded DataStore (provides id_lookup_new).
        out_path: Where to persist the stage output.
    """
    model = build_llm_client()
    prompts = read_json("prompts/prompts_eval_translate.json")
    exceptions: list = []
    counter = 0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        for pid in ranking:
            for rec in list(ranking[pid].keys())[:TOP_K_0]:
                fut = executor.submit(
                    grade_ranking_entry,
                    model, prompts, ranking, pid, rec,
                    ds.get_project_desc(pid),
                    ds.id_lookup_new, exceptions,
                )
                futures[fut] = (pid, rec)

        for fut in concurrent.futures.as_completed(futures):
            counter += 1
            if counter % 500 == 0:
                write_json(ranking, out_path)
                print(f"[L_1] checkpoint: {counter} records graded …")
            if counter % 10 == 0:
                print(f"[L_1] graded {counter} records …")

    write_json(ranking, out_path)
    print(f"[L_1] Saved → {out_path}  |  exceptions: {len(exceptions)}")
    return ranking


def validate_grades(ranking: dict, top_k: int = TOP_K_0) -> None:
    """Warn about any record within top_k that is missing a grade."""
    for pid in ranking:
        for rec in ranking[pid]:
            try:
                pos = int(rec.replace("r", ""))
                if pos < top_k and "grade" not in ranking[pid][rec]:
                    print(f"Missing grade: project={pid}, rec={rec}")
            except ValueError:
                pass


# ── Stage L_2: Re-ranking ─────────────────────────────────────────────────────

def run_stage_l2(ranking: dict, out_path: str = "L_2.json") -> dict:
    """Re-rank every project's results by LLM grade."""
    reranked = rerank_by_grade(ranking, top_k=TOP_K_0, top_k_out=TOP_K)
    write_json(reranked, out_path)
    print(f"[L_2] Saved → {out_path}")
    return reranked


# ── Stage L_3: Snippet summarization ─────────────────────────────────────────

def run_stage_l3(
    ranking: dict,
    out_path: str = "L_3.json",
) -> dict:
    """
    Summarise snippets and inject markdown links for every project in the ranking.

    Projects are parallelised at the outer level; each project also parallelises
    its own per-record summarisation calls internally.

    Args:
        ranking:  Output of run_stage_l2.
        out_path: Where to persist the stage output.
    """
    model = build_llm_client()

    def _summarise_project(pid: str) -> None:
        with concurrent.futures.ThreadPoolExecutor() as inner:
            futs = [
                inner.submit(set_snippet_summary, model, ranking, pid, rec)
                for rec in ranking[pid]
            ]
            concurrent.futures.wait(futs)

        for rec in ranking[pid]:
            ranking[pid][rec]["snippet_summ_mkd"] = convert_to_markdown(ranking[pid][rec])
            ranking[pid][rec]["snippet_summ_mkd"] = prepend_url(ranking[pid][rec])

    with concurrent.futures.ThreadPoolExecutor() as executor:
        pids = list(ranking.keys())
        futs = {executor.submit(_summarise_project, pid): pid for pid in pids}
        for i, fut in enumerate(concurrent.futures.as_completed(futs), start=1):
            pid = futs[fut]
            print(f"[L_3] {i}/{len(pids)}  project={pid} done")

    write_json(ranking, out_path)
    print(f"[L_3] Saved → {out_path}")
    return ranking


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the batch ranking pipeline.")
    parser.add_argument(
        "--pids",
        nargs="*",
        default=None,
        metavar="PROJECT_ID",
        help="Optional list of project ids to process. Defaults to all projects.",
    )
    parser.add_argument("--l0", default="L_0.json", help="Output path for stage L_0.")
    parser.add_argument("--l1", default="L_1.json", help="Output path for stage L_1.")
    parser.add_argument("--l2", default="L_2.json", help="Output path for stage L_2.")
    parser.add_argument("--l3", default="L_3.json", help="Output path for stage L_3.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()

    print("Loading data …")
    ds = DataStore()

    ranking_l0 = run_stage_l0(ds, project_ids=args.pids, out_path=args.l0)
    ranking_l1 = run_stage_l1(ranking_l0, ds, out_path=args.l1)
    validate_grades(ranking_l1)
    ranking_l2 = run_stage_l2(ranking_l1, out_path=args.l2)
    ranking_l3 = run_stage_l3(ranking_l2, out_path=args.l3)

    # Print a sample from the first project
    sample_pid = list(ranking_l3.keys())[0]
    print(f"\nSample output — project: {sample_pid}")
    print("=" * 60)
    for rec, data in ranking_l3[sample_pid].items():
        print(data.get("snippet_summ_mkd", ""))
        print("-" * 50)