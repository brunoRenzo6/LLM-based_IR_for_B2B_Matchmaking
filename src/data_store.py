"""
data_store.py - Loads and caches all shared runtime data into a single DataStore object.

Usage:
    from data_store import DataStore
    ds = DataStore()
"""

from typing import Dict, List, Optional

import numpy as np
import polars as pl

import config
from io_utils import (
    load_projects_description,
    load_project_to_doc,
    load_project_chunks,
    load_u_chunk_id_list,
    load_doc_embeddings,
    load_key_embeddings,
    load_id_lookup,
    load_corpus_chunks,
)


UI_PROJECT_ID = "ui_query"  # sentinel project_id used when querying from the UI


class DataStore:
    """Single source of truth for all pre-loaded data and derived mappings."""

    def __init__(self):
        self._load_projects()
        self._load_items()
        self._build_aligned_chunks()

    @classmethod
    def from_query(cls, query: str) -> "DataStore":
        """
        Build a minimal DataStore from a free-text query string (e.g. from the UI).

        The project side is stubbed with a single mock project keyed by
        UI_PROJECT_ID. The item/seller corpus is loaded normally so all
        vector search and grading logic works identically to batch mode.
        """
        ds = cls.__new__(cls)
        ds._stub_project(query)
        ds._load_items()
        ds._build_aligned_chunks()
        return ds

    def _stub_project(self, query: str) -> None:
        """Populate project-side attributes from a single free-text query."""
        import pandas as pd

        pid = UI_PROJECT_ID
        self.df_projects = pd.DataFrame([{"id": pid, "project_desc": query}])
        self.projects_chunks_d = {"__ui_doc__": {"0": query}}
        self.project_to_doc = {pid: "__ui_doc__"}
        self.doc_to_project = {"__ui_doc__": pid}
        self.chunk_to_doc = {query: ["__ui_doc__"]}
        # No cached embedding — embeddings.py will fall back to a live API call.
        self.emb_dict: Dict[str, List[float]] = {}

    # ── Project data ──────────────────────────────────────────────────────────

    def _load_projects(self):
        self.df_projects = load_projects_description(config.DB_USER, config.USER_FILE)

        self.projects_chunks_d = load_project_chunks(config.DB_USER, config.TABLE_USER)
        self.project_to_doc = load_project_to_doc(config.DB_USER, config.TABLE_USER)
        self.doc_to_project = {v: k for k, v in self.project_to_doc.items()}

        # chunk_text → [doc_id, ...]
        self.chunk_to_doc: Dict[str, List[str]] = {}
        for did, chunks in self.projects_chunks_d.items():
            chunk = chunks["0"]
            self.chunk_to_doc.setdefault(chunk, []).append(did)

        # Pre-built embedding lookup: project_id → embedding vector
        df_key_emb = load_key_embeddings(config.SYSTEM, config.DB_USER, config.TABLE_USER, config.MODE_USER)
        self.emb_dict: Dict[str, List[float]] = dict(
            zip(df_key_emb["project_id"], df_key_emb["embedding"])
        )

    # ── Item / document data ──────────────────────────────────────────────────

    def _load_items(self):
        self.u_chunk_id_l = load_u_chunk_id_list(
            config.SYSTEM, config.DB_ITEM, config.TABLE_ITEM, config.MODE_ITEM
        )
        self.doc_emb: np.ndarray = load_doc_embeddings(
            config.SYSTEM, config.DB_ITEM, config.TABLE_ITEM, config.EMB_MODEL, config.MODE_ITEM
        )
        self.id_lookup_new = load_id_lookup(config.SYSTEM, config.DB_ITEM, config.TABLE_ITEM)

    # ── Aligned chunk dataframe ───────────────────────────────────────────────

    def _build_aligned_chunks(self):
        """Pre-align the chunk DataFrame to exactly match the numpy doc_emb array."""
        df_chunks_o = load_corpus_chunks(
            config.SYSTEM, config.DB_ITEM, config.TABLE_ITEM, config.MODE_ITEM
        )
        df_chunks_o = (
            df_chunks_o
            .group_by(["root_doc_id", "chunk"])
            .agg([
                pl.col("u_chunk_id").first(),
                pl.col("doc_id").first(),
            ])
        )
        df_order = (
            pl.DataFrame({"u_chunk_id": self.u_chunk_id_l})
            .with_row_index("order_idx")
        )
        self.df_chunks_aligned: pl.DataFrame = (
            df_order
            .join(df_chunks_o, on="u_chunk_id", how="left")
            .sort("order_idx")
            .drop("order_idx")
        )

    # ── Convenience helpers ───────────────────────────────────────────────────

    def get_stored_embedding(self, pid_l: List[str]) -> Optional[List[float]]:
        """Return cached embedding for the first matching project id."""
        for pid in pid_l:
            if pid in self.emb_dict:
                return self.emb_dict[pid]
        return None

    def get_project_chunks(self, project_id: str) -> List[str]:
        doc_id = self.project_to_doc[project_id]
        return list(self.projects_chunks_d[doc_id].values())

    def get_project_desc(self, project_id: str) -> str:
        row = self.df_projects[self.df_projects["id"] == project_id]
        return row["project_desc"].iloc[0]