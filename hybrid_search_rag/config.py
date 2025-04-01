# hybrid_search_rag/config.py
"""Configuration settings for the project."""
import os
from dotenv import load_dotenv
import logging
# Removed google.generativeai import as it's likely handled in llm_interface.py

# Determine project root assuming config.py is in hybrid_search_rag package
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load .env file from the project root
dotenv_path = os.path.join(project_root, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    logging.info(f"Loaded environment variables from: {dotenv_path}")
else:
    logging.warning(f".env file not found at {dotenv_path}. API keys or Project ID might be missing.")

# --- API Keys & Cloud Config ---
# Commented out unused keys
# HF_API_TOKEN = os.getenv("HF_API_TOKEN")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Google API Key (Used for Gemini LLM via llm_interface)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- NEW: Vertex AI Configuration ---
# Load from .env file preferably
VERTEX_AI_PROJECT = os.getenv("VERTEX_AI_PROJECT") # Your Google Cloud Project ID
VERTEX_AI_LOCATION = os.getenv("VERTEX_AI_LOCATION", "us-central1") # Your GCP region, e.g., us-central1

# --- Model Settings ---
# Embedding model (Semantic Search) - NOW USING VERTEX AI
# EMBEDDING_MODEL_NAME = 'BAAI/bge-large-en-v1.5' # Previous HF model
EMBEDDING_MODEL_NAME = "models/embedding-001" # Vertex AI Model ID (check for latest recommended version)

# --- LLM Configuration ---
LLM_PROVIDER = "google"                   # Set provider to Google
# LLM_MODEL_ID = "llama3-70b-8192"        # Groq model (example)
# LLM_MODEL_ID = "gemma2-9b-it"           # Groq model (example)
LLM_MODEL_ID = "gemini-2.0-flash"         # Set to a valid Gemini model ID


# --- Data Settings (Unchanged) ---
# Points to the 'data' directory at the project root
DATA_DIR = os.path.join(project_root, "data")
METADATA_FILE = "combined_metadata.json"
EMBEDDINGS_FILE = "combined_embeddings.npy" # This will store Vertex AI embeddings now
BM25_INDEX_FILE = "bm25_index.pkl"

# --- arXiv Fetcher Settings (Unchanged) ---
DEFAULT_ARXIV_QUERY = "Reinforcement Learning"
MAX_ARXIV_RESULTS = 100 # Default number of papers to fetch

# --- Web Fetcher Settings (Unchanged) ---
TARGET_WEB_URLS = [
    "https://distill.pub/",
    "https://developers.google.com/machine-learning/",
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://mistral.ai/news/mixtral-of-experts/",
    "https://huggingface.co/blog/mixtral",
    "https://jalammar.github.io/illustrated-transformer/",
    "https://nlp.seas.harvard.edu/papers/",
    "https://icml.cc/virtual/2024/papers.html",
    "https://icml.cc/virtual/2023/papers.html",
    "https://icml.cc/virtual/2022/papers.html",
    "https://icml.cc/virtual/2021/papers.html",
    "https://icml.cc/virtual/2020/papers.html",
    "https://icml.cc/virtual/2019/papers.html",
    "https://icml.cc/virtual/",
    "https://nips.cc/virtual/2024/papers.html?layout=detail",
    "https://nips.cc/virtual/2023/papers.html?layout=detail",
    "https://nips.cc/virtual/2022/papers.html?layout=detail",
    "https://nips.cc/virtual/2021/papers.html?layout=detail",
    "https://nips.cc/virtual/2020/papers.html?layout=detail",
    "https://nips.cc/virtual/2019/papers.html?layout=detail",
    "https://nips.cc/virtual/",
    # Add other relevant HTML or PDF URLs here for your full dataset
]

# --- Recommender Settings (Unchanged, but performance might vary with new embeddings) ---
DEFAULT_QUERY = "Explain recurrent neural networks"
TOP_N_RESULTS = 10 # Default final number of results to display in CLI fallback
RANK_FUSION_K = 60 # Constant for Reciprocal Rank Fusion (RRF)
SEMANTIC_CANDIDATES = 150 # Number of candidates for semantic search phase
KEYWORD_CANDIDATES = 150 # Number of candidates for keyword search phase

# --- RAG Settings (Unchanged) ---
RAG_NUM_DOCS = 13 # Number of retrieved docs to use as context for LLM
MAX_CONTEXT_LENGTH_PER_DOC = 2000 # Max characters per doc snippet
LLM_MAX_NEW_TOKENS = 4096 # Max tokens for the LLM response
LLM_TEMPERATURE = 0.3 # LLM creativity
LLM_API_TIMEOUT = 500 # Timeout in seconds for API calls

# --- NEW/UPDATED: Vertex AI Specific Settings ---
VERTEX_AI_BATCH_LIMIT = 250 # Batch size limit for Vertex AI embedding requests (check docs for your model)
# Optional: Add delay between Vertex AI batches if needed, though often not required
# EMBEDDING_API_DELAY = 0.1 # Delay in seconds

# --- Utility (Unchanged) ---
# Ensure data directory exists early
try:
    os.makedirs(DATA_DIR, exist_ok=True)
except OSError as e:
    logging.error(f"Could not create data directory {DATA_DIR}: {e}")

# --- API Key / Project ID Checks ---
# Check only if specific provider is selected

# Check for Google LLM API Key
if LLM_PROVIDER == "google" and not GOOGLE_API_KEY:
    logging.warning("Google API Key (GOOGLE_API_KEY) not found for selected LLM provider.")
    print("\n" + "="*50)
    print("Warning: Google API Key (GOOGLE_API_KEY) not found for LLM.")
    print(f"Please create or check the .env file in the project root ({project_root}) and add:")
    print("GOOGLE_API_KEY=YourActualGoogleKeyValue")
    print("You can get a key from Google AI Studio (https://ai.google.dev/) or Google Cloud Console.")
    print("="*50 + "\n")

# Check for Vertex AI Project ID (needed for embeddings)
if not VERTEX_AI_PROJECT:
    logging.warning("Google Cloud Project ID (VERTEX_AI_PROJECT) not found.")
    print("\n" + "="*50)
    print("Warning: Google Cloud Project ID (VERTEX_AI_PROJECT) not found.")
    print(f"Please create or check the .env file in the project root ({project_root}) and add:")
    print("VERTEX_AI_PROJECT=your-gcp-project-id")
    print("Ensure you have also authenticated via `gcloud auth application-default login`.")
    print("="*50 + "\n")
    # Consider exiting if required:
    # import sys
    # sys.exit("Vertex AI Project ID is missing. Exiting.")

