# Ranking System – Module Reference

## Overview

The monolithic notebook has been split into focused, single-responsibility modules:

```
LLM-based_IR_for_B2B_Matchmaking/
├── config.py               # All constants and environment variables
├── io_utils.py             # File I/O (JSON, CSV, pickle, numpy, Polars)
├── data_store.py           # Loads & caches all shared runtime data
├── embeddings.py           # Cached + live Azure OpenAI embedding retrieval
├── ranking.py              # Vector search, cut-off, aggregation, multi-query fusion
├── llm_eval.py             # LLM relevance grading chain
├── reranker.py             # Grade-based re-ranking
├── snippet_summarizer.py   # LLM snippet summarization + markdown link injection
├── pipeline.py             # End-to-end orchestration (L_0 → L_1 → L_2 → L_3)
└── streamlit_app.py        # quick app to demo the ranking system
```

## Pipeline Stages

| Stage | File output | What happens |
|-------|-------------|--------------|
| **L_0** | `L_0.json` | Multi-query vector search → top-`TOP_K_0` structured results |
| **L_1** | `L_1.json` | LLM grades each of the `TOP_K_0` retrieved results for relevance |
| **L_2** | `L_2.json` | Results re-ordered by LLM grade, trimmed to top `TOP_K` |
| **L_3** | `L_3.json` | Snippet summaries generated; doc IDs replaced with markdown links |

## Usage

```bash
cd LLM-based_IR_for_B2B_Matchmaking
python pipeline.py
```

Or import individual stages in a notebook:

```python
from data_store import DataStore
from ranking import get_rankings_multi_query, build_struct_sellers
from llm_eval import build_llm_client, grade_ranking_entry

ds = DataStore()
df_chunks, df_docs, df_root_docs = get_rankings_multi_query(["query here"], ds)
```

## Configuration

All tuneable parameters live in `config.py`. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODE_ITEM` | `"train"` | `"train"` or `"test"` — selects data files |
| `MODE_USER` | `"test"` | `"train"` or `"test"` — selects data files |
| `EMB_MODEL` | `"text-embedding-3-small"` | Azure OpenAI embedding model |
| `TOP_K_0` | `15` | Number of sellers retrieved and graded (pre-reranking pool, L_0 → L_1) |
| `TOP_K` | `10` | Number of sellers ultimately presented to the buyer (post-reranking, L_2 output) |
| `SNIPPET_TOP_K` | `15` | Chunks per result in the snippet |
| `MAX_CHUNK_CHARS` | `500` | Truncation limit for chunk text |

## Key Design Decisions

- **`DataStore`** centralises all data loading, so modules never load files independently. Pass the `ds` instance around rather than re-loading.
- **`embeddings.py`** uses an O(1) dict lookup first; only falls back to a live API call on a cache miss.
- **Concurrency** – L_1 and L_3 use `ThreadPoolExecutor` for I/O-bound LLM calls.
- **Idempotency** – the grader skips records that already have a `grade` key, making retries safe.