# -*- coding: utf-8 -*-
"""
Configuration settings for the Hybrid Search RAG project.

Organizes settings into logical sections for better readability and management.
Handles environment variables for sensitive information like API keys.
Includes options for LLM fallback across multiple models within providers,
and across different providers (e.g., Google -> Groq).
"""

import os
import logging
import sys # Import sys for stderr printing
from typing import List, Optional, Final # Added Final for constants

# --- Basic Logging Setup (Configure early) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-7s - [Config] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)

# --- Project Root Determination ---
try:
    PROJECT_ROOT: Final[str] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logger.info(f"Project root determined: {PROJECT_ROOT}")
except Exception as e:
    logger.error(f"Failed to determine project root: {e}", exc_info=True)
    PROJECT_ROOT: Final[str] = os.getcwd() # Fallback
    logger.warning(f"Falling back to current working directory as project root: {PROJECT_ROOT}")


# --- Environment Variable Loading (.env) ---
DOTENV_PATH: Final[str] = os.path.join(PROJECT_ROOT, '.env')
try:
    from dotenv import load_dotenv
    if os.path.exists(DOTENV_PATH):
        load_dotenv(dotenv_path=DOTENV_PATH)
        logger.info(f"Loaded environment variables from: {DOTENV_PATH}")
    else:
        logger.warning(f".env file not found at {DOTENV_PATH}. API keys might be missing.")
except ImportError:
    logger.warning("`python-dotenv` package not found. Cannot load .env file.")
except Exception as e:
    logger.error(f"Error loading .env file from {DOTENV_PATH}: {e}", exc_info=True)


# ==============================================================================
# --- Core Paths ---
# ==============================================================================
DATA_DIR: Final[str] = os.path.join(PROJECT_ROOT, "data")
METADATA_FILE: Final[str] = "combined_metadata.json"
EMBEDDINGS_FILE: Final[str] = "combined_embeddings.npy"
BM25_INDEX_FILE: Final[str] = "bm25_index.pkl"


# ==============================================================================
# --- API Keys & Cloud Configuration ---
# ==============================================================================
# Load API keys from environment variables (set in .env file).
GOOGLE_API_KEY: Final[Optional[str]] = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY: Final[Optional[str]] = os.getenv("GROQ_API_KEY")
# Add other keys if needed (e.g., HF_API_TOKEN)


# ==============================================================================
# --- LLM Configuration ---
# ==============================================================================
# --- Provider Order ---
# Define the order in which to try providers. The first provider is primary.
LLM_PROVIDER_ORDER: Final[List[str]] = ["google", "groq"] # Example: Try Google first, then Groq

# --- Models per Provider ---
# Define lists of model IDs to try *within* each provider, in order of preference.
# The interface will iterate through these models for the first provider in
# LLM_PROVIDER_ORDER, then move to the models for the second provider, etc.
LLM_GOOGLE_MODELS: Final[List[str]] = [
    "gemini-2.5-flash-preview-04-17", # Primary Google choice
    "gemini-2.5-pro-preview-03-25",   # Fallback Google choice
    "gemini-2.0-flash"                # Older alias
]
LLM_GROQ_MODELS: Final[List[str]] = [
    "meta-llama/llama-4-maverick-17b-128e-instruct",    # Primary Groq choice
    "meta-llama/llama-4-scout-17b-16e-instruct",        # Fallback Groq choice
]

# --- General LLM Parameters ---
# These parameters will be attempted for all API calls.
LLM_TEMPERATURE: Final[float] = 0.3
LLM_MAX_NEW_TOKENS: Final[int] = 4096
LLM_API_TIMEOUT: Final[int] = 60 # Timeout in seconds (adjust as needed)


# ==============================================================================
# --- Embedding Model Configuration ---
# ==============================================================================
EMBEDDING_MODEL_NAME: Final[str] = 'models/text-embedding-004'
EMBEDDING_DIM: Final[int] = 768
VERTEX_AI_BATCH_LIMIT: Final[int] = 100


# ==============================================================================
# --- Data Source Settings ---
# ==============================================================================
DEFAULT_ARXIV_QUERY: Final[str] = ""
MAX_ARXIV_RESULTS: Final[int] = 100
TARGET_WEB_URLS: Final[List[str]] = [
    "https://distill.pub/",
]
MAX_PAGES_TO_CRAWL: Final[int] = 200
CRAWL_DELAY_SECONDS: Final[float] = 0.5
ALLOWED_DOMAINS: Final[List[str]] = []
FETCH_TIMEOUT: Final[int] = 45
HEAD_TIMEOUT: Final[int] = 20


# ==============================================================================
# --- Retrieval & Ranking Settings ---
# ==============================================================================
DEFAULT_QUERY: Final[str] = "Explain Retrieval-Augmented Generation (RAG)"
TOP_N_RESULTS: Final[int] = 10
SEMANTIC_CANDIDATES: Final[int] = 150
KEYWORD_CANDIDATES: Final[int] = 150
RANK_FUSION_K: Final[int] = 60


# ==============================================================================
# --- RAG (Retrieval-Augmented Generation) Settings ---
# ==============================================================================
RAG_NUM_DOCS: Final[int] = 15
MAX_CONTEXT_LENGTH_PER_DOC: Final[int] = 2000


# ==============================================================================
# --- Utility & Checks ---
# ==============================================================================
try:
    os.makedirs(DATA_DIR, exist_ok=True)
    logger.info(f"Data directory checked/created: {DATA_DIR}")
except OSError as e:
    logger.error(f"Could not create data directory {DATA_DIR}: {e}", exc_info=True)
except Exception as e:
    logger.error(f"Unexpected error checking/creating data directory {DATA_DIR}: {e}", exc_info=True)

# --- API Key / Configuration Checks ---
def run_config_checks():
    """Runs checks for essential configuration and logs warnings."""
    config_ok = True
    providers_checked = set()
    at_least_one_provider_ok = False
    logger.info("Running configuration checks...")

    if not LLM_PROVIDER_ORDER:
        logger.error("LLM_PROVIDER_ORDER list is empty. No LLM providers specified.")
        config_ok = False

    for provider in LLM_PROVIDER_ORDER:
        if provider in providers_checked: continue # Avoid duplicate checks
        providers_checked.add(provider)
        provider_models = []
        api_key = None
        key_name = ""
        provider_ok = False

        if provider == "google":
            provider_models = LLM_GOOGLE_MODELS
            api_key = GOOGLE_API_KEY
            key_name = "GOOGLE_API_KEY"
        elif provider == "groq":
            provider_models = LLM_GROQ_MODELS
            api_key = GROQ_API_KEY
            key_name = "GROQ_API_KEY"
        else:
            logger.error(f"Unsupported provider found in LLM_PROVIDER_ORDER: '{provider}'.")
            config_ok = False
            continue # Check next provider

        if not provider_models:
             logger.warning(f"Provider '{provider}' is in order, but its model list (e.g., LLM_{provider.upper()}_MODELS) is empty.")
             # Don't mark config as failed, but it won't be used.
        elif not api_key:
             logger.warning(f"Provider '{provider}' is configured with models, but its API key ({key_name}) is missing in .env.")
             # Mark config as failed only if this is the *only* provider or the first one.
             if len(LLM_PROVIDER_ORDER) == 1 or provider == LLM_PROVIDER_ORDER[0]:
                  config_ok = False
        else:
             logger.info(f"Provider '{provider}' seems configured (API key found, models listed: {provider_models}).")
             provider_ok = True
             at_least_one_provider_ok = True

    # General Checks
    if EMBEDDING_DIM <= 0:
         logger.error(f"Invalid EMBEDDING_DIM ({EMBEDDING_DIM}). Must be positive.")
         config_ok = False

    if not at_least_one_provider_ok:
        logger.critical("No LLM providers appear to be configured correctly with both models and API keys.")
        config_ok = False # Ensure overall config status reflects this critical issue

    if config_ok:
         logger.info(f"Configuration checks passed (at least one provider seems usable).")
    else:
         logger.critical("One or more critical configuration checks failed. Functionality will be impaired. Please review errors/warnings and check your .env file and provider model lists.")
         # Print a prominent message if critical config is missing
         print("\n" + "="*60 + "\n"
               "🚨 CRITICAL CONFIGURATION ERROR 🚨\n"
               "   No LLM provider seems fully configured with both models and an API key,\n"
               "   or an essential setting (like EMBEDDING_DIM) is invalid.\n"
               f"   Please check logs above and ensure the required keys (e.g., GOOGLE_API_KEY, GROQ_API_KEY)\n"
               f"   are correctly set in '{DOTENV_PATH}' and that model lists (e.g., LLM_GOOGLE_MODELS)\n"
               "   are populated for the providers listed in LLM_PROVIDER_ORDER.\n"
               "   The application may not function correctly.\n"
               + "="*60 + "\n", file=sys.stderr)

# Run checks when the module is imported
run_config_checks()

logger.info("Configuration loading complete.")
