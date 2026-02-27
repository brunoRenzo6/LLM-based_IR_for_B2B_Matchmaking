"""
embeddings.py - Embedding retrieval (cached or live via Azure OpenAI).
"""

import os
from typing import List

from langchain_openai import AzureOpenAIEmbeddings

import config
from data_store import DataStore


def get_embedding_for_query(query: str, ds: DataStore) -> List[float]:
    """
    Return an embedding vector for the query string.

    Strategy:
      1. Look up the query text in the project chunk index to get a project id.
      2. Return the pre-computed embedding for that project (O(1) dict lookup).
      3. Fall back to a live Azure OpenAI call only when no cached match exists.
    """
    if query in ds.chunk_to_doc:
        did_l = ds.chunk_to_doc[query]
        pid_l = [ds.doc_to_project[did] for did in did_l]
        cached = ds.get_stored_embedding(pid_l)
        if cached is not None:
            return cached

    print("get_embedding_for_query(): calling Azure OpenAI (cache miss)")
    return _call_azure_embedding(query)


def _call_azure_embedding(text: str) -> List[float]:
    embeddings = AzureOpenAIEmbeddings(
        model=config.EMB_MODEL,
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )
    return embeddings.embed_query(text)
