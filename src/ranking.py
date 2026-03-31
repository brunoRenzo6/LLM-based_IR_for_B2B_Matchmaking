"""
ranking.py - Vector search, cut-off selection, and multi-query ranking logic.
"""

from typing import Dict, List, Tuple

import numpy as np
import polars as pl
from sklearn.metrics.pairwise import cosine_similarity

from data_store import DataStore
from embeddings import get_embedding_for_query


# ── Core vector search ────────────────────────────────────────────────────────

def vector_search(query: str, ds: DataStore) -> pl.DataFrame:
    """Compute cosine similarities between the query and every item chunk."""
    query_emb = np.array(get_embedding_for_query(query, ds)).reshape(1, -1)
    cos_sim = cosine_similarity(query_emb, ds.doc_emb)[0]

    return ds.df_chunks_aligned.with_columns(pl.Series("cos_sim", cos_sim))


# ── Cut-off selection ─────────────────────────────────────────────────────────

def get_lower_cut(
    df_chunks_rank: pl.DataFrame,
    mode: str = "zscore",
    z_threshold: float = 3.5,
    margin: float = 0.1,
    min_unique_docs: int = 15,
) -> float:
    """
    Iteratively relax the threshold until at least `min_unique_docs` distinct
    root documents survive the cut.
    """
    df = df_chunks_rank.with_columns(
        cos_sim_zscore=(
            (pl.col("cos_sim") - pl.col("cos_sim").mean()) / pl.col("cos_sim").std()
        )
    )

    n_unique = 0
    while n_unique < min_unique_docs:
        if mode == "zscore":
            filtered = df.filter(pl.col("cos_sim_zscore") > z_threshold)
            lower_cut = filtered.select(pl.col("cos_sim").min()).item()
        else:  # margin
            max_sim = df.select(pl.col("cos_sim").max()).item()
            lower_cut = max_sim - margin
            filtered = df.filter(pl.col("cos_sim") > lower_cut)

        n_unique = filtered.select(pl.col("root_doc_id").n_unique()).item()
        z_threshold -= 0.25
        margin += 0.05

    return lower_cut


# ── Aggregation helpers ───────────────────────────────────────────────────────

def apply_lower_cut(df_chunks_rank: pl.DataFrame, lower_cut: float) -> pl.DataFrame:
    return (
        df_chunks_rank
        .with_columns(
            pl.when(pl.col("cos_sim") >= lower_cut)
            .then(pl.col("cos_sim"))
            .otherwise(0)
            .alias("cos_sim")
        )
        .select(["u_chunk_id", "doc_id", "root_doc_id", "cos_sim"])
    )


def aggregate_to_docs(df_chunks: pl.DataFrame, ratio: bool = False) -> pl.DataFrame:
    regularization = 100
    df = df_chunks.group_by(["doc_id", "root_doc_id"]).agg([
        pl.col("cos_sim").sum().alias("cos_sim_sum"),
        pl.len().alias("n_chunks"),
    ])

    score_expr = (
        (pl.col("cos_sim_sum") / (pl.col("n_chunks") + regularization))
        if ratio
        else pl.col("cos_sim_sum")
    )
    return df.with_columns(score_expr.alias("cos_sim_ratio")).sort("cos_sim_ratio", descending=True)


def aggregate_to_root_docs(df_docs: pl.DataFrame, ratio: bool = False) -> pl.DataFrame:
    regularization = 100
    df = df_docs.group_by("root_doc_id").agg([
        pl.col("cos_sim_sum").sum(),
        pl.col("n_chunks").sum(),
    ])

    score_expr = (
        (pl.col("cos_sim_sum") / (pl.col("n_chunks") + regularization))
        if ratio
        else pl.col("cos_sim_sum")
    )
    return df.with_columns(score_expr.alias("cos_sim_ratio")).sort("cos_sim_ratio", descending=True)


# ── Single-query pipeline ─────────────────────────────────────────────────────

def get_rankings(
    query: str,
    ds: DataStore,
    weight: float = 1.0,
    ratio: bool = False,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Returns (df_chunks_rank, df_chunks_rank_cut, df_docs_rank, df_root_docs_rank).
    """
    df_chunks_rank = vector_search(query, ds)
    lower_cut = get_lower_cut(df_chunks_rank, mode="zscore", z_threshold=3.0)

    df_chunks_rank_cut = apply_lower_cut(df_chunks_rank, lower_cut)
    if weight != 1.0:
        df_chunks_rank_cut = df_chunks_rank_cut.with_columns(pl.col("cos_sim") * weight)

    df_docs_rank = aggregate_to_docs(df_chunks_rank_cut, ratio)
    df_root_docs_rank = aggregate_to_root_docs(df_docs_rank, ratio)

    return df_chunks_rank, df_chunks_rank_cut, df_docs_rank, df_root_docs_rank


# ── Multi-query pipeline ──────────────────────────────────────────────────────

def _add_doc_and_root_columns(df: pl.DataFrame, id_lookup: dict) -> pl.DataFrame:
    df = df.with_columns(pl.col("u_chunk_id").str.split("_").list.get(0).alias("doc_id"))
    lookup_df = pl.DataFrame(
        [{"doc_id": k, "root_doc_id": str(v["root_doc_id"])} for k, v in id_lookup.items()]
    )
    df = df.join(lookup_df, on="doc_id", how="left")

    cols = [c for c in df.columns if c not in ["doc_id", "root_doc_id"]]
    return df.select([cols[0]] + ["doc_id", "root_doc_id"] + cols[1:])


def _coalesce_nulls(df: pl.DataFrame, score_n: int) -> pl.DataFrame:
    for i in range(score_n + 1):
        df = df.with_columns(
            pl.coalesce(pl.col(f"score_{i}"), 0).alias(f"score_{i}"),
            pl.coalesce(pl.col("score_sum"), 0).alias("score_sum"),
        )
    return df


def _join_and_sum(rank_list: List[pl.DataFrame], key_col: str) -> pl.DataFrame:
    dfs = [df.clone() for df in rank_list]

    for i, _ in enumerate(dfs):
        if i == 0:
            dfs[i] = dfs[i].rename({"score": "score_0"})
        else:
            dfs[i] = dfs[i].select([key_col, pl.col("score").alias(f"score_{i}")])

    result = dfs[0].with_columns(pl.col("score_0").alias("score_sum"))
    for i, df_i in enumerate(dfs[1:], start=1):
        result = result.join(df_i, on=key_col, how="full")
        result = result.with_columns(
            pl.coalesce(pl.col(key_col), pl.col(f"{key_col}_right")).alias(key_col)
        )
        result = _coalesce_nulls(result, i)
        result = result.with_columns(
            (pl.col("score_sum") + pl.col(f"score_{i}")).alias("score_sum")
        )
        result = result.drop([f"{key_col}_right"])

    result = result.sort("score_sum", descending=True)
    cols = result.columns
    cols.remove("score_sum")
    return result.select(cols + ["score_sum"])


def get_rankings_multi_query(
    queries: List[str],
    ds: DataStore,
    ratio: bool = False,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Run the ranking pipeline for multiple queries and fuse the results.
    Returns (df_chunks_rank, df_docs_rank, df_root_docs_rank).
    """
    cut_dfs = []
    for query in queries:
        _, df_cut, _, _ = get_rankings(query, ds, ratio=ratio)
        cut_dfs.append(df_cut.select(["u_chunk_id", pl.col("cos_sim").alias("score")]))

    df_fused = _join_and_sum(cut_dfs, key_col="u_chunk_id")
    df_fused = _add_doc_and_root_columns(df_fused, ds.id_lookup_new)

    df_chunks = df_fused.select(
        ["u_chunk_id", "doc_id", "root_doc_id", pl.col("score_sum").alias("cos_sim")]
    )
    df_docs = aggregate_to_docs(df_chunks, ratio)
    df_root_docs = aggregate_to_root_docs(df_docs, ratio)

    return df_chunks, df_docs, df_root_docs


# ── Result structuring ────────────────────────────────────────────────────────

def build_struct_sellers(
    df_chunks_rank: pl.DataFrame,
    df_root_docs_rank: pl.DataFrame,
    top_k: int = 15,
    snippet_top_k: int = 15,
) -> dict:
    """Package the top-ranked root documents and their best matching chunks.

    Args:
        df_chunks_rank:    Chunk-level ranked dataframe from the retrieval stage.
        df_root_docs_rank: Root-document-level ranked dataframe.
        top_k:             Number of root docs to include (pass ``TOP_K_0`` when
                           the output feeds the reranker, or ``TOP_K`` for a
                           single-stage ranking without reranking).
        snippet_top_k:     Max chunks per seller kept for the snippet.
    """
    struct_sellers = {}

    for i, row in enumerate(df_root_docs_rank.head(top_k).iter_rows(named=True)):
        if row["cos_sim_ratio"] == 0:
            continue

        rdid = row["root_doc_id"]
        top_chunks = (
            df_chunks_rank
            .filter(pl.col("root_doc_id") == rdid)
            .top_k(snippet_top_k, by="cos_sim")
        )

        chunks = {
            f"c{j}": {"u_chunk_id": c["u_chunk_id"], "cos_sim": c["cos_sim"]}
            for j, c in enumerate(top_chunks.iter_rows(named=True))
        }

        struct_sellers[f"r{i}"] = {
            "root_doc_id": rdid,
            "cos_sim_ratio": row["cos_sim_ratio"],
            "chunks": chunks,
        }

    return struct_sellers