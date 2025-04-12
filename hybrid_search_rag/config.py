# hybrid_search_rag/config.py
"""Configuration settings for the project."""
import os
from dotenv import load_dotenv
import logging
# Removed google.generativeai import as it's likely handled in llm_interface.py

# --- Determine project root ---
# This assumes this config.py file lives inside a 'hybrid_search_rag' folder,
# and the main project folder is one level up. Smart! Keeps paths relative.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Load .env file ---
# This looks for a file named '.env' in the main project folder.
# That file is where we (should!) secretly store API keys and other sensitive stuff.
# DO NOT COMMIT THE .env file to Git! Seriously. Don't. 🙅‍♂️
dotenv_path = os.path.join(project_root, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    logging.info(f"Loaded environment variables from: {dotenv_path}")
else:
    # If the .env file is missing, things might break later (like API calls). Warn the user!
    logging.warning(f".env file not found at {dotenv_path}. API keys or Project ID might be missing.")

# --- API Keys & Cloud Config ---
# Reading API keys and project IDs from the environment (loaded from .env).
# Commented out keys we aren't using right now (like Hugging Face or Groq).
# HF_API_TOKEN = os.getenv("HF_API_TOKEN")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Google API Key (Used for the Gemini LLM - the brain of our RAG!)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- NEW: Vertex AI Configuration ---
# We're using Vertex AI for embeddings now. Need the Google Cloud Project ID and location.
# VERTEX_AI_PROJECT = os.getenv("VERTEX_AI_PROJECT") # Your Google Cloud Project ID
# VERTEX_AI_LOCATION = os.getenv("VERTEX_AI_LOCATION", "us-central1") # Your GCP region, default 'us-central1' seems reasonable.

# --- Model Settings ---
# Embedding model (Semantic Search) - Specifies WHICH model to use for turning text into vectors.
# Previously used a Hugging Face model, now switched to Google's Vertex AI model.
# EMBEDDING_MODEL_NAME = 'BAAI/bge-large-en-v1.5' # Old one
EMBEDDING_MODEL_NAME = "models/embedding-001" # Vertex AI Model ID. Check Google's docs for the latest

# --- LLM Configuration ---
# Which LLM provider and model are we using to generate the final answers?
LLM_PROVIDER = "google"                   # Set provider to Google (Gemini)
# LLM_MODEL_ID = "llama3-70b-8192"        # Example if using Groq
# LLM_MODEL_ID = "gemma2-9b-it"           # Another Groq example
LLM_MODEL_ID = "gemini-2.0-flash"         # Current choice: A fast Gemini model. Make sure it's a valid ID!

# --- Data Settings ---
# Where should we store our downloaded data, generated embeddings, etc.?
# Points to a 'data' folder inside the main project directory.
DATA_DIR = os.path.join(project_root, "data")
# Filenames for the important stuff we save.
METADATA_FILE = "combined_metadata.json"    # Info about each document chunk
EMBEDDINGS_FILE = "combined_embeddings.npy" # The vector embeddings (will be Vertex AI format now)
BM25_INDEX_FILE = "bm25_index.pkl"          # The keyword search index

# --- arXiv Fetcher Settings ---
# Settings for grabbing papers from arXiv.
DEFAULT_ARXIV_QUERY = "" # What to search for if the user doesn't specify.
MAX_ARXIV_RESULTS = 100 # How many papers to grab by default. Don't go too crazy!

# --- Web Fetcher / Crawler Settings ---
# The list of starting URLs for our web crawler. It will try to grab content from these
# and follow links (within limits) to find more.
TARGET_WEB_URLS = [
    "https://distill.pub/", # High-quality ML articles
    # "https://developers.google.com/machine-learning/", # Google's ML docs
    # "https://nlp.seas.harvard.edu/papers/", # Harvard NLP papers page
    # # # Adding recent ICML / NeurIPS proceedings pages - good source of ML papers!
    # "https://icml.cc/virtual/2024/papers.html?layout=detail",
    # "https://www.jmlr.org/",
    # "https://icml.cc/virtual/2023/papers.html",
    # "https://icml.cc/virtual/2022/papers.html",
    # "https://icml.cc/virtual/2021/papers.html",
    # "https://icml.cc/virtual/2020/papers.html",
    # "https://icml.cc/virtual/2019/papers.html",
    # "https://nips.cc/virtual/2024/papers.html?layout=detail",
    # "https://nips.cc/virtual/2023/papers.html?layout=detail",
    # "https://nips.cc/virtual/2022/papers.html?layout=detail",
    # "https://nips.cc/virtual/2021/papers.html?layout=detail",
    # "https://nips.cc/virtual/2020/papers.html?layout=detail",
    # "https://nips.cc/virtual/2019/papers.html?layout=detail",
    # "https://quotes.toscrape.com/", # Example site for testing
    # Add MORE high-quality, relevant URLs here! Think key blogs, labs, open-source docs.
]

# --- Recommender Settings ---
MAX_PAGES_TO_CRAWL = 200 # (Limit the total number of pages fetched per run)
CRAWL_DELAY_SECONDS = 0.2 # Politeness delay between requests
ALLOWED_DOMAINS = [] # Optional: If empty, allows same domain as start URL...

# Settings for how the hybrid search (semantic + keyword) works.
DEFAULT_QUERY = "Explain recurrent neural networks" # Default question for the CLI/app.
TOP_N_RESULTS = 10 # How many final results to show the user in the CLI fallback (if LLM fails).
RANK_FUSION_K = 60 # Magic number for Reciprocal Rank Fusion (RRF). Balances scores. 60 is common.
SEMANTIC_CANDIDATES = 150 # How many candidates to get from the embedding (vector) search initially.
KEYWORD_CANDIDATES = 150 # How many candidates to get from the keyword (BM25) search initially. More candidates -> potentially better recall, but slower fusion.

# --- RAG Settings ---
# Settings for the Retrieval-Augmented Generation part (feeding context to the LLM).
RAG_NUM_DOCS = 15 # How many of the top *fused* document chunks to actually send to the LLM as context. Balance between enough info and context window limits.
MAX_CONTEXT_LENGTH_PER_DOC = 2000 # Max characters to take from each chunk. Keeps context size manageable.
LLM_MAX_NEW_TOKENS = 4096 # Max number of tokens the LLM is allowed to generate in its response. Prevents runaways.
LLM_TEMPERATURE = 0.3 # Controls LLM creativity/randomness. Lower = more focused, higher = more creative (and potentially weird). 0.3 is quite focused.
LLM_API_TIMEOUT = 500 # How long to wait (in seconds) for the LLM API to respond before giving up. Generous timeout needed for long responses.
FETCH_TIMEOUT = 30
HEAD_TIMEOUT = 15

# --- NEW/UPDATED: Vertex AI Specific Settings ---
# How many texts can we send to the Vertex AI embedding API in one go? Check the model docs! 250 is often safe for text-embedding-gecko.
VERTEX_AI_BATCH_LIMIT = 350
# Optional delay between batches (in seconds) if we hit rate limits. Often not needed for Vertex unless hammering it.
# EMBEDDING_API_DELAY = 0.1

# --- Utility ---
# Make sure the data directory exists right when this config is loaded. No excuses!
try:
    os.makedirs(DATA_DIR, exist_ok=True)
except OSError as e:
    logging.error(f"Could not create data directory {DATA_DIR}: {e}") # Log error if creation fails.

# --- API Key / Project ID Checks ---
# Do some checks right here to warn the user LOUDLY if essential keys/IDs are missing based on selected providers.
# Saves debugging headaches later! 🎉

# Check for Google LLM API Key (only if 'google' provider is selected for LLM)
if LLM_PROVIDER == "google" and not GOOGLE_API_KEY:
    logging.warning("Google API Key (GOOGLE_API_KEY) not found for selected LLM provider.")
    # Print a big, noticeable warning to the console.
    print("\\n" + "="*50)
    print("Warning: Google API Key (GOOGLE_API_KEY) not found for LLM.")
    print(f"Please create or check the .env file in the project root ({project_root}) and add:")
    print("GOOGLE_API_KEY=YourActualGoogleKeyValue")
    print("You can get a key from Google AI Studio (https://ai.google.dev/) or Google Cloud Console.")
    print("="*50 + "\\n")