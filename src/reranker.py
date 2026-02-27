"""
reranker.py - Re-rank retrieved results using LLM grades.
"""

import copy


def rerank_by_grade(ranking: dict, top_k: int = 15, top_k_out: int = 10) -> dict:
    """
    Sort each project's results by their LLM grade (descending) and keep only
    the top ``top_k_out`` results for presentation to the buyer.

    Args:
        ranking:    Output of the grading stage (L_1 JSON).
        top_k:      How many records per project to consider when re-ranking
                    (should match TOP_K_0 — the full pre-reranking pool).
        top_k_out:  How many records to retain in the output
                    (should match TOP_K — the final presented set).

    Returns:
        A new dict with the same structure but keys ``rr0``, ``rr1``, … ordered
        by grade and capped at ``top_k_out`` entries per project.
    """
    reranked: dict = {}

    for pid, project_data in ranking.items():
        reranked[pid] = {}

        candidate_keys = [k for k in list(project_data.keys())[:top_k]]
        scored = [
            (project_data[rec].get("grade", 0), rec)
            for rec in candidate_keys
        ]
        scored_sorted = sorted(scored, reverse=True, key=lambda t: t[0])

        for i, (_, rec) in enumerate(scored_sorted[:top_k_out]):
            reranked[pid][f"rr{i}"] = copy.deepcopy(project_data[rec])

    return reranked