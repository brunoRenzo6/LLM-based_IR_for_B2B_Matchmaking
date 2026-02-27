"""
io_utils.py - File I/O helpers (JSON, CSV, pickle, numpy).
"""

import json
import os
import numpy as np
import pandas as pd
import polars as pl

from config import SYSTEM, SYSTEM_ROOT


# ── JSON ──────────────────────────────────────────────────────────────────────

def read_json(f_path: str) -> dict:
    with open(f_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: dict, f_path: str) -> None:
    with open(f_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def file_exists(f_path: str) -> bool:
    return os.path.isfile(f_path)


# ── Domain-specific loaders ───────────────────────────────────────────────────

def load_projects_description(db_user: str, user_file: str) -> pd.DataFrame:
    path = f"{SYSTEM_ROOT}/{SYSTEM}/document_warehouse/{db_user}/{user_file}"
    return pd.read_csv(path)


def load_project_to_doc(db_user: str, table_user: str) -> dict:
    path = f"{SYSTEM_ROOT}/{SYSTEM}/document_warehouse/{db_user}/{table_user}/project_to_doc.json"
    return read_json(path)


def load_project_chunks(db_user: str, table_user: str) -> dict:
    path = f"{SYSTEM_ROOT}/{SYSTEM}/document_warehouse/{db_user}/{table_user}/{table_user}.json"
    return read_json(path)


def load_u_chunk_id_list(system: str, db_item: str, table_item: str, mode: str) -> list:
    path = f"{SYSTEM_ROOT}/{system}/document_warehouse/{db_item}/{table_item}/u_chunk_id_l_{mode}.json"
    return read_json(path)


def load_doc_embeddings(system: str, db_item: str, table_item: str, emb_model: str, mode: str) -> np.ndarray:
    path = f"{SYSTEM_ROOT}/{system}/vector_warehouse/{db_item}/{table_item}/{emb_model}/doc_emb_{mode}.npy"
    return np.load(path)


def load_key_embeddings(system: str, db_user: str, table_user: str, mode: str) -> pd.DataFrame:
    path = f"{SYSTEM_ROOT}/{system}/vector_warehouse/{db_user}/{table_user}/df_key_emb_{mode}.pkl"
    return pd.read_pickle(path)


def load_id_lookup(system: str, db_item: str, table_item: str) -> dict:
    path = f"{SYSTEM_ROOT}/{system}/document_warehouse/{db_item}/{table_item}/id_lookup_new.json"
    return read_json(path)


def load_corpus_chunks(system: str, db_item: str, table_item: str, mode: str) -> pl.DataFrame:
    path = f"{SYSTEM_ROOT}/{system}/document_warehouse/{db_item}/{table_item}/corpus_chunks_{mode}.csv"
    return pl.read_csv(path, infer_schema_length=10000).cast(pl.String)


def load_chunk_document(system: str, db_item: str, table_item: str, doc_id: str) -> dict:
    path = f"{SYSTEM_ROOT}/{system}/document_warehouse/{db_item}/{table_item}/documents/{doc_id}.json"
    return read_json(path)
