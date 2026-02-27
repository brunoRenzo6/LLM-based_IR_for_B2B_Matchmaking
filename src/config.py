"""
config.py - Central configuration for the ranking system.
"""

import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# --- Run Mode ---
MODE_USER = "train"
MODE_ITEM = "test"

# --- System Paths ---
SYSTEM = "system2"
SYSTEM_ROOT = "../../../.."

# --- Database / Table References ---
DB_USER = "2024_10_19_from_originals"
TABLE_USER = "table_project_originals_03"
USER_FILE = f"projects_o_{MODE_USER}.csv"

DB_ITEM = "website_documents_20250305"
TABLE_ITEM = "item_chunks_br"

# --- Embedding Model ---
EMB_MODEL = "text-embedding-3-small"

# --- Azure OpenAI Credentials ---
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

if not AZURE_API_KEY or not AZURE_ENDPOINT:
    raise ValueError("Missing Azure credentials. Make sure AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are set in your .env file.")

os.environ["AZURE_OPENAI_API_KEY"] = AZURE_API_KEY
os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_ENDPOINT

# --- LLM ---
GPT_DEPLOYMENT = "gpt-4o-mini"
GPT_API_VERSION = "2024-02-15-preview"
GPT_TEMPERATURE = 0.1

# --- Ranking ---
TOP_K_0 = 15          # Number of sellers retrieved and graded (pre-reranking pool)
TOP_K = 10            # Number of sellers ultimately presented to the buyer (post-reranking)
SNIPPET_TOP_K = 15    # Number of chunks per seller used to build the snippet
MAX_CHUNK_CHARS = 500

# --- Translation ---
# Set to True to translate project descriptions and seller snippets to English.
# Set to False to keep all content in the original Brazilian Portuguese.
# (This config is only used to be abe to actually demo the project in english)
TRANSLATE_TO_ENGLISH = True