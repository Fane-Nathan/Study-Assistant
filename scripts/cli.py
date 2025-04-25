# -*- coding: utf-8 -*-
"""
Main script to run the CLI application with hybrid search and RAG.
Default mode: Strict RAG (answers only from local docs).
--general mode: Hybrid-Knowledge RAG (uses local docs + LLM internal knowledge).
"""

import argparse
import logging
import ssl
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow INFO messages
import sys
import arxiv
import nltk
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union, Generator # Added Generator
import textwrap
import traceback
import asyncio

# --- Logger Setup ---
# Configure logger specifically for this module
# Use basicConfig with force=True to ensure it applies even if root logger was configured elsewhere
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-7s - [CLI] %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True)
logger = logging.getLogger(__name__) # Use module name

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---


# --- Project Module Imports ---
try:
    from hybrid_search_rag import config
    # Ensure resource_fetcher is imported correctly
    from hybrid_search_rag.data.resource_fetcher import fetch_arxiv_papers, crawl_and_fetch_web_articles
    from hybrid_search_rag.retrieval.embedding_model_gemini import EmbeddingModel
    from hybrid_search_rag.data.data_manager import DataManager
    from hybrid_search_rag.retrieval.recommender import Recommender, NltkManager, RecommendationParams
    from rank_bm25 import BM25Okapi
    # Import the updated llm_interface
    from hybrid_search_rag.llm.llm_interface import get_llm_response, get_llm_response_stream
    # Highlighting is assumed removed based on previous steps
except ImportError as e:
    # Use print for critical import errors as logger might not be fully configured yet
    print(f"ERROR: Failed to import project modules in cli.py: {e}", file=sys.stderr)
    print(f"Ensure you are running from the project root or the path setup is correct.", file=sys.stderr)
    print(f"Current sys.path: {sys.path}", file=sys.stderr)
    sys.exit(1)
# --- End Project Module Imports ---


# --- NLTK Data Check ---
# Flag to ensure check/download runs only once per script execution
_nltk_data_checked_cli = False

def check_nltk_data():
    """
    Checks required NLTK data ('punkt', 'stopwords').
    Attempts download if missing. Exits if download fails.
    Ensures check runs only once per script execution.
    """
    global _nltk_data_checked_cli
    if _nltk_data_checked_cli:
        # logger.debug("NLTK data check already performed in this CLI execution.")
        return # Already checked in this run

    required_data = {'punkt': 'tokenizers/punkt', 'stopwords': 'corpora/stopwords'}
    missing_data = []
    logger.info("Checking required NLTK data...")
    for name, path in required_data.items():
        try:
            nltk.data.find(path)
            logger.info(f"NLTK data '{name}' found.")
        except LookupError:
            logger.warning(f"NLTK data '{name}' not found.")
            missing_data.append(name) # Add to list if missing

    if missing_data:
        logger.warning(f"Missing NLTK data packages: {', '.join(missing_data)}.")
        # Use print for CLI visibility during download attempt
        print(f"\nAttempting to download missing NLTK data: {', '.join(missing_data)}...", file=sys.stderr)
        download_success = True
        try:
            # Attempt to bypass SSL verification if needed (common issue)
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass # Doesn't exist, proceed normally
            else:
                ssl._create_default_https_context = _create_unverified_https_context
                logger.info("Applied SSL context workaround for NLTK download.")

            for name in missing_data:
                print(f"Downloading NLTK package: {name}...")
                # Use quiet=False for CLI to show progress/errors
                if nltk.download(name, quiet=False):
                    logger.info(f"Successfully downloaded NLTK data '{name}'.")
                    # Verify immediately after download
                    try:
                        nltk.data.find(required_data[name])
                        logger.info(f"Verified NLTK data '{name}' after download.")
                    except LookupError:
                        logger.error(f"Verification failed after downloading '{name}'. Download might be incomplete or corrupted.")
                        download_success = False
                        break # Stop trying if verification fails
                else:
                    # nltk.download returns None if download failed
                    logger.error(f"NLTK download command failed for '{name}'. Check network connection or NLTK server status.")
                    download_success = False
                    break # Stop trying if download fails

        except Exception as e:
            logger.error(f"An error occurred during NLTK download: {e}", exc_info=True)
            download_success = False

        if not download_success:
            print("\nAutomatic NLTK download failed.", file=sys.stderr)
            print("Please try installing the data manually in your environment:", file=sys.stderr)
            print(">>> import nltk", file=sys.stderr)
            for name in missing_data:
                print(f">>> nltk.download('{name}')", file=sys.stderr)
            print("\nExiting due to missing NLTK data.", file=sys.stderr)
            sys.exit(1) # Exit if essential data couldn't be obtained
        else:
            logger.info("Finished NLTK download attempt.")
    else:
        logger.info("All required NLTK data packages found.")

    _nltk_data_checked_cli = True # Mark as checked for this execution


# --- Text Chunking ---
def chunk_text_by_sentences(text: str, sentences_per_chunk: int = 5, overlap_sentences: int = 1) -> List[str]:
    """Splits text into chunks by sentences with overlap."""
    if not text: return []
    try:
        # Ensure punkt is available before tokenizing
        nltk.data.find('tokenizers/punkt') # This will raise LookupError if check_nltk_data failed
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        logger.critical("NLTK 'punkt' tokenizer not found during chunking. Cannot proceed. Ensure NLTK data was downloaded.")
        # Re-raise or return empty list depending on desired handling
        raise # Let the calling function handle the critical error
    except Exception as e:
        logger.warning(f"Sentence tokenization failed with unexpected error: {e}. Falling back to newline split.")
        sentences = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not sentences: sentences = [p.strip() for p in text.split('\n') if p.strip()]

    if not sentences:
        logger.warning("Could not extract sentences or lines for chunking.")
        return []

    chunks = []
    start_index = 0
    while start_index < len(sentences):
        end_index = min(start_index + sentences_per_chunk, len(sentences))
        chunk_sentences = sentences[start_index:end_index]
        chunks.append(" ".join(chunk_sentences))
        step = max(1, sentences_per_chunk - overlap_sentences)
        start_index += step
    return chunks
# --- End Text Chunking ---

# --- Global Variables for Loaded Components ---
loaded_metadata: Optional[List[Dict[str, Any]]] = None
loaded_embeddings: Optional[np.ndarray] = None
loaded_bm25_index: Optional[BM25Okapi] = None
loaded_recommender: Optional[Recommender] = None

# --- Component Loading ---
def load_components(force_reload: bool = False):
    """Loads data (metadata, embeddings, index) and initializes recommender."""
    global loaded_metadata, loaded_embeddings, loaded_bm25_index, loaded_recommender
    if force_reload or loaded_metadata is None or loaded_recommender is None:
        logger.info(f"{'Forcing reload' if force_reload else 'Loading core components'}...")
        data_manager = DataManager(config.DATA_DIR, config.METADATA_FILE, config.EMBEDDINGS_FILE, config.BM25_INDEX_FILE)
        loaded_metadata, loaded_embeddings, loaded_bm25_index = data_manager.load_all_data()

        if loaded_metadata is None:
            logger.error("Metadata loading failed. Cannot initialize recommender.")
            raise RuntimeError("Metadata loading failed. Run the 'fetch' command first.")

        try:
            # Ensure NLTK data needed by Recommender/NltkManager is checked/loaded
            check_nltk_data() # Ensure data is ready before Recommender init

            embedder = EmbeddingModel(config.EMBEDDING_MODEL_NAME)
            loaded_recommender = Recommender(embed_model=embedder)
            logger.info("Recommender initialized.")
        except SystemExit: # Catch exit from check_nltk_data
             logger.critical("NLTK data check failed during component loading. Cannot initialize recommender.")
             raise RuntimeError("Failed to acquire NLTK data during component loading.")
        except Exception as e:
            logger.error(f"Recommender initialization failed: {e}", exc_info=True)
            loaded_recommender = None
            raise RuntimeError("Recommender initialization failed.") from e

        if loaded_embeddings is None: logger.warning("Embeddings file not found or invalid; semantic search disabled.")
        if loaded_bm25_index is None: logger.warning("BM25 index file not found or invalid; keyword search disabled.")
        if loaded_embeddings is None and loaded_bm25_index is None:
            logger.error("Both embeddings and BM25 index are missing. Search functionality severely limited.")

# --- Data Fetching Logic ---
def setup_data_and_fetch(args: argparse.Namespace) -> str:
    """Handles the 'fetch' command: gets documents, chunks, embeds, indexes, and saves."""
    logger.info("Starting data fetch and processing...")
    status_messages = []

    # --- Ensure NLTK data is available before proceeding ---
    try:
        check_nltk_data() # Verify NLTK data specifically for the fetch process
    except SystemExit: # Catch exit if NLTK download fails
        return "Error: Failed to acquire necessary NLTK data ('punkt', 'stopwords'). Cannot proceed with fetch."
    # ---

    # Determine Sources
    arxiv_query = ""; target_urls = []; max_results = args.num_arxiv
    try:
        if args.suggest_sources:
            logger.info(f"Attempting LLM source suggestion for topic: '{args.topic}'")
            if not args.topic: return "Error: Topic (-t) required for --suggest-sources."
            suggestion_prompt = f"""Suggest search sources for the topic "{args.topic}":
1. Two arXiv query strings. Prefix *exactly* with "ARXIV_QUERY: ".
2. Five high-quality, relevant URLs (e.g., conference proceedings, key labs, tech blogs). Prefix *exactly* with "URL: ".
Provide *only* the queries and URLs in the specified format."""
            logger.info("Requesting source suggestions from LLM...")
            try:
                # Use the non-streaming interface for suggestions
                suggestion_response = get_llm_response(suggestion_prompt)
                if suggestion_response:
                    suggested_arxiv_queries = [line.replace("ARXIV_QUERY:", "").strip() for line in suggestion_response.splitlines() if line.strip().startswith("ARXIV_QUERY:")]
                    suggested_urls = [line.replace("URL:", "").strip() for line in suggestion_response.splitlines() if line.strip().startswith("URL:")]
                    arxiv_query = suggested_arxiv_queries[0] if suggested_arxiv_queries else ""
                    target_urls = suggested_urls if suggested_urls else []
                    if not arxiv_query: logger.warning("LLM did not suggest arXiv queries."); max_results = 0
                    if not target_urls: logger.warning("LLM did not suggest URLs.")
                    logger.info(f"Using LLM suggestions: arXiv='{arxiv_query}', URLs={len(target_urls)}")
                else:
                     logger.error("LLM suggestion failed (empty response). Clearing sources.")
                     arxiv_query = ""; max_results = 0; target_urls = []
                     status_messages.append("Warning: LLM suggestion failed.")
            except Exception as llm_e:
                 logger.error(f"Error during LLM suggestion API call: {llm_e}. Clearing sources.", exc_info=True)
                 arxiv_query = ""; max_results = 0; target_urls = []
                 status_messages.append(f"Warning: LLM suggestion failed ({llm_e}).")
        elif args.arxiv_query:
            arxiv_query = args.arxiv_query
            target_urls = config.TARGET_WEB_URLS if hasattr(config, 'TARGET_WEB_URLS') else [] # Use config or empty list
            logger.info(f"Using custom arXiv query: '{arxiv_query}' and {'default' if target_urls else 'no'} web URLs.")
        else:
            arxiv_query = config.DEFAULT_ARXIV_QUERY if hasattr(config, 'DEFAULT_ARXIV_QUERY') else ""
            target_urls = config.TARGET_WEB_URLS if hasattr(config, 'TARGET_WEB_URLS') else []
            logger.info("Using default arXiv query and web URLs from config (if defined).")
    except Exception as e:
         logger.error(f"Error determining sources: {e}", exc_info=True)
         return f"Error deciding sources: {e}"

    # Fetching Documents
    logger.info(f"Fetching: arXiv query='{arxiv_query}' (max={max_results}), URLs={len(target_urls)}")
    arxiv_metadata_list = fetch_arxiv_papers(arxiv_query, max_results) if (arxiv_query and max_results > 0) else []
    if target_urls:
        logger.info("Running asynchronous web fetcher...")
        try:
            # Handle asyncio loop management carefully
            try:
                loop = asyncio.get_event_loop_policy().get_event_loop()
                if loop.is_running():
                    logger.debug("Event loop already running, using asyncio.run() for web fetch.")
                    web_metadata_list = asyncio.run(crawl_and_fetch_web_articles(target_urls))
                else:
                    logger.debug("No running event loop, using loop.run_until_complete() for web fetch.")
                    web_metadata_list = loop.run_until_complete(crawl_and_fetch_web_articles(target_urls))
            except RuntimeError as e:
                 logger.warning(f"Asyncio loop management issue ({e}), falling back to simple asyncio.run().")
                 web_metadata_list = asyncio.run(crawl_and_fetch_web_articles(target_urls))
        except Exception as web_e:
             logger.error(f"Web crawling failed: {web_e}", exc_info=True)
             status_messages.append(f"Warning: Web crawling failed ({web_e}).")
             web_metadata_list = []

        logger.info(f"Web fetcher finished, got {len(web_metadata_list)} results.")
    else:
        web_metadata_list = []
    original_documents = arxiv_metadata_list + web_metadata_list
    num_docs_fetched = len(original_documents)
    if num_docs_fetched == 0: return "Error: No data fetched from any source."
    logger.info(f"Fetched {num_docs_fetched} total documents.")

    # Chunking Documents
    logger.info("Chunking documents...")
    all_chunk_metadata = []
    for doc_index, doc_meta in enumerate(original_documents):
        original_content = doc_meta.get('content', '')
        original_url = doc_meta.get('url', f'doc_{doc_index}')
        if not original_content or not original_content.strip(): continue
        try:
            text_chunks = chunk_text_by_sentences(original_content)
        except Exception as chunk_e: # Catch potential errors from chunking
             logger.error(f"Failed to chunk document {original_url}: {chunk_e}", exc_info=True)
             continue # Skip this document if chunking fails

        if not text_chunks: continue
        for chunk_index, chunk_text in enumerate(text_chunks):
            all_chunk_metadata.append({
                "chunk_id": f"{original_url}_chunk_{chunk_index}", "chunk_text": chunk_text,
                "original_url": original_url, "original_title": doc_meta.get('title', 'Untitled'),
                "source": doc_meta.get('source', 'unknown'), "authors": doc_meta.get('authors', []),
                "published": doc_meta.get('published', None),
                "arxiv_entry_id": doc_meta.get('entry_id') if doc_meta.get('source') == 'arxiv' else None
            })
    num_chunks = len(all_chunk_metadata)
    if num_chunks == 0: return "Error: No text chunks generated from documents."
    logger.info(f"Generated {num_chunks} chunks.")

    # Processing & Saving Chunks
    logger.info("Processing chunks (embedding & indexing)...")
    manager = DataManager(config.DATA_DIR, config.METADATA_FILE, config.EMBEDDINGS_FILE, config.BM25_INDEX_FILE)
    embedder = EmbeddingModel(config.EMBEDDING_MODEL_NAME)
    all_chunk_texts = [chunk['chunk_text'] for chunk in all_chunk_metadata]
    embeddings = None; bm25_index = None; embed_success = False; bm25_success = False

    try: # Embedding
        embeddings = embedder.encode(all_chunk_texts, task_type="RETRIEVAL_DOCUMENT")
        if embeddings is not None and isinstance(embeddings, np.ndarray) and embeddings.shape[0] == num_chunks:
            logger.info(f"Embeddings generated (Shape: {embeddings.shape}).")
            embed_success = True
        else: logger.error("Embedding generation failed or returned unexpected result.")
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}", exc_info=True)
        status_messages.append("Warning: Embedding failed.")

    try: # BM25 Indexing
        # NltkManager should be ready now due to earlier check
        tokenized_corpus = [NltkManager.tokenize_text(t) for t in all_chunk_texts if isinstance(t, str) and t.strip()]
        tokenized_corpus = [tok for tok in tokenized_corpus if tok] # Remove empty lists after tokenization
        if tokenized_corpus:
            bm25_index = BM25Okapi(tokenized_corpus)
            logger.info("BM25 index built.")
            bm25_success = True
        else:
            logger.warning("No valid tokens found for BM25 index after processing chunks.")
            if not NltkManager.NLTK_DATA_AVAILABLE['punkt']:
                 status_messages.append("Warning: BM25 indexing skipped (NLTK 'punkt' unavailable).")

    except Exception as e:
        logger.error(f"BM25 index building failed: {e}", exc_info=True)
        status_messages.append("Warning: BM25 indexing failed.")

    try: # Saving Data
        manager.save_all_data(all_chunk_metadata, embeddings, bm25_index)
        logger.info("Processed data saved successfully.")
        load_components(force_reload=True) # Reload components with new data
        success_msg = f"Success: Fetched {num_docs_fetched} docs, generated {num_chunks} chunks. Saved metadata"
        if embed_success: success_msg += ", embeddings"
        if bm25_success: success_msg += ", BM25 index"
        success_msg += "."
        if status_messages: success_msg += " " + " ".join(status_messages)
        return success_msg
    except Exception as e:
        logger.error(f"Failed to save processed data: {e}", exc_info=True)
        return f"Error: Failed to save data. Details: {e}"

# --- Fallback Result Formatting ---
def format_fallback_results(results: List[Tuple[Dict[str, Any], float]], num_to_show: int) -> str:
    """Formats retrieval results into a string if the LLM fails."""
    if not results: return "LLM response failed; no relevant document chunks found."
    actual_num_to_show = min(num_to_show, len(results))
    lines = [f"LLM response failed. Top {actual_num_to_show} retrieved document chunks (fallback):"]
    for rank, (chunk_meta, score) in enumerate(results[:actual_num_to_show]):
        lines.append("-" * 20)
        lines.append(f"{rank + 1}. [Score: {score:.4f}] [Src: {chunk_meta.get('source', '?')}]")
        lines.append(f"   Title: {chunk_meta.get('original_title', 'N/A')}")
        lines.append(f"   URL: {chunk_meta.get('original_url', '#')}")
        lines.append(f"   Snippet: {(chunk_meta.get('chunk_text', '') or '')[:200]}...")
    return "\n".join(lines)

# --- Recommendation Logic ---
# Adjusted return type hint to match implementation
def run_recommendation(query: str, num_final_results: int, general_mode: bool, concise_mode: bool) -> Tuple[Generator[str, None, None], List[str], List[Dict[str, Any]]]:
    """
    Handles recommendation: retrieves chunks, prepares context, calls LLM stream.
    Returns a tuple: (response_generator, formatted_source_list, raw_context_chunks_list)
    """
    mode_name = "Hybrid" if general_mode else "Strict"
    style_name = "Concise" if concise_mode else "Detailed"
    logger.info(f"Running recommendation ({mode_name} RAG, {style_name} Prompt) for query: '{query[:100]}...'")

    response_generator: Generator[str, None, None] # Type hint for generator
    formatted_source_list: List[str] = []
    raw_context_chunks_list: List[Dict[str, Any]] = [] # Still needed for return signature
    hybrid_results: List[Tuple[Dict[str, Any], float]] = []
    captured_exception = None # Variable to hold exception if needed

    try:
        # 1. Load Components
        if loaded_recommender is None or loaded_metadata is None:
             logger.info("Components not loaded, attempting load...")
             load_components() # This will run check_nltk_data again if needed
        if loaded_recommender is None or loaded_metadata is None:
             raise RuntimeError("Core components failed to load. Run 'fetch' command first.")

        # 2. Retrieval Step
        logger.info("Retrieving relevant document chunks...")
        num_candidates = max(config.RAG_NUM_DOCS + 5, num_final_results + 5)
        # Ensure num_candidates doesn't exceed available metadata
        num_candidates = min(num_candidates, len(loaded_metadata))
        if num_candidates > 0:
            rec_params = RecommendationParams(
                semantic_candidates=config.SEMANTIC_CANDIDATES,
                keyword_candidates=config.KEYWORD_CANDIDATES,
                fusion_k=config.RANK_FUSION_K,
                top_n_final=num_candidates
            )
            hybrid_results = loaded_recommender.recommend(
                query=query,
                resource_metadata=loaded_metadata,
                resource_embeddings=loaded_embeddings,
                bm25_index=loaded_bm25_index,
                params=rec_params
            )
            logger.info(f"Retrieved {len(hybrid_results)} candidate chunks.")
        else: logger.warning("Skipping retrieval (no metadata available or zero candidates needed).")

        # 3. Prepare Context, Formatted Sources, AND Raw Chunks
        context_string = "No relevant document chunks were found in the local data."
        reference_map = {}
        if hybrid_results:
            top_chunks_for_rag = hybrid_results[:config.RAG_NUM_DOCS]
            # Store raw chunk dictionaries (still needed for the return tuple)
            raw_context_chunks_list = [chunk_meta for chunk_meta, _ in top_chunks_for_rag]
            context_texts = []
            for i, chunk_meta in enumerate(raw_context_chunks_list):
                 chunk_num = i + 1
                 title = chunk_meta.get('original_title', 'N/A')
                 url = chunk_meta.get('original_url', '#')
                 snippet = (chunk_meta.get('chunk_text', '') or '')[:config.MAX_CONTEXT_LENGTH_PER_DOC]
                 # Adjusted context format slightly for clarity
                 context_texts.append(f"Source [{chunk_num}]:\nTitle: {title}\nURL: {url}\nContent Snippet:\n{snippet}")
                 reference_map[chunk_num] = f"{title} (URL: {url})"
            context_string = "\n\n---\n\n".join(context_texts)
            # Create the formatted list for display
            formatted_source_list = [f"[{num}] {details}" for num, details in reference_map.items()]
            logger.info(f"Prepared context from {len(raw_context_chunks_list)} chunks.")
        else:
            logger.info("No relevant document chunks found to provide as context.")
            formatted_source_list = ["No relevant document chunks found."]

        # 4. Prompt Selection (Simplified without explicit citation instructions)
        final_prompt: str = ""
        prompt_template_base = """[INST] {system_message}

Provided Document Context:
{context}

User Question: {query} [/INST]
Answer:"""

        if general_mode:
            if concise_mode:
                system_message = ( # Hybrid + Concise (No Citation)
                    "You are a helpful AI research assistant."
                    "Provide a structured summary answering the user's question."
                    "Use the provided document context for key details, supplementing with your general knowledge where appropriate."
                    "**Format your entire response using standard Markdown.**"
                    "Use the following structure:"
                    "## Summary\n[Provide a brief overview answering the main question.]\n\n"
                    "## Key Details\n[List the most important findings or points using bullet points (* or -). Refer to information from the context.]\n\n"
                    "## Conclusion\n[Summarize the main takeaways.]"
                )
            else:
                system_message = ( # Hybrid + Detailed (No Citation)
                     "You are an AI assistant expert at analyzing technical documents."
                     "Provide a **detailed and comprehensive** answer to the user's question."
                     "Base your answer primarily on the 'Provided Document Context', but integrate relevant general knowledge smoothly for context and clarity."
                     "**Structure your response clearly using standard Markdown:**"
                     "**## Overview:**\n[Start with a concise paragraph summarizing the main answer.]\n\n"
                     "**## Detailed Analysis:**\n[Use ### Sub-Headings for distinct topics. Under each sub-heading, explain the topic thoroughly using multiple sentences or bullet points (* or -). Synthesize information across different context sources where applicable.]\n\n"
                     "**## Conclusion:**\n[Provide a concluding paragraph summarizing the key points and any limitations based on the provided context.]"
                )
        else: # Strict Mode
            if concise_mode:
                system_message = ( # Strict + Concise (No Citation)
                     "You are an AI assistant expert at analyzing technical documents."
                     "Provide a concise summary answering the user's question using **only** information found in the 'Provided Document Context'. **Do not use any outside knowledge.**"
                     "**Format your entire response using standard Markdown.**"
                     "Use the following structure:"
                     "## Summary\n[Provide a brief overview answering the main question based *only* on the context.]\n\n"
                     "## Key Findings\n[List the most important findings or points using bullet points (* or -). Extract information directly from the context.]\n\n"
                     "## Conclusion\n[Summarize the main takeaways based *only* on the context. State if the provided context is insufficient to fully answer.]"
                )
            else:
                system_message = ( # Strict + Detailed (No Citation)
                     "You are an AI assistant expert at analyzing technical documents."
                     "Provide a detailed answer using **only** information found in the 'Provided Document Context'. **Do not use any outside knowledge.**"
                     "**Structure your response clearly using standard Markdown:**"
                     "**## Overview:**\n[Start with a concise paragraph summarizing the main answer based *only* on the context.]\n\n"
                     "**## Detailed Analysis:**\n[Use ### Sub-Headings for distinct topics found *only* in the documents. Under each sub-heading, explain the topic thoroughly using multiple sentences or bullet points (* or -). Synthesize information across different context sources where applicable, but stick strictly to the provided text.]\n\n"
                     "**## Conclusion:**\n[Provide a concluding paragraph summarizing the key points based *only* on the context. Explicitly state if the information in the context is limited or insufficient.]"
                )

        final_prompt = prompt_template_base.format(system_message=system_message, context=context_string, query=query)

        # --- REMOVED Redundant Log Line ---
        # logger.info(f"Sending STREAMING request to LLM ({config.LLM_PROVIDER}, {config.LLM_MODEL_ID})...")
        logger.info("Preparing to call LLM for response generation...") # More general log
        response_generator = get_llm_response_stream(final_prompt)

    except RuntimeError as e:
        logger.error(f"Recommendation failed due to component loading error: {e}", exc_info=False)
        captured_exception = e # Capture exception
        # --- FIXED: Pass captured exception to error_generator ---
        def error_generator(err): yield f"Error: {err}. Cannot run recommendation."
        response_generator = error_generator(captured_exception)
        formatted_source_list = []
        raw_context_chunks_list = [] # Ensure it's empty on error
    except Exception as e:
        logger.error(f"Unexpected error during recommendation setup: {e}", exc_info=True)
        captured_exception = e # Capture exception
        # --- FIXED: Pass captured exception to error_generator ---
        def error_generator(err): yield f"An unexpected error occurred during setup: {err}"
        response_generator = error_generator(captured_exception)
        formatted_source_list = []
        raw_context_chunks_list = [] # Ensure it's empty on error

    # Return the generator, formatted sources, and raw chunks
    return response_generator, formatted_source_list, raw_context_chunks_list


# --- arXiv Search Logic ---
def run_arxiv_search(query: str, num_results: int) -> List[Dict[str, Any]]:
    """Performs a direct arXiv search."""
    logger.info(f"Searching arXiv directly for: '{query}' (Top {num_results})")
    results_list: List[Dict[str, Any]] = []
    try:
        client = arxiv.Client(page_size=min(num_results, 100), delay_seconds=1.0, num_retries=3)
        search = arxiv.Search(query=query, max_results=num_results, sort_by=arxiv.SortCriterion.Relevance)
        for result in client.results(search):
            results_list.append({
                "title": result.title or "N/A",
                "authors": [str(a) for a in result.authors],
                "published": result.published.strftime('%Y-%m-%d') if result.published else "N/A",
                "pdf_url": result.pdf_url or 'N/A',
                "summary": (result.summary or '').replace('\n', ' ').strip() # Clean summary
            })
        logger.info(f"arXiv search found {len(results_list)} results.")
    except Exception as e:
        logger.error(f"arXiv search failed: {e}", exc_info=True)
    return results_list


# --- Main CLI Execution ---
def main():
    """Parses arguments and runs the chosen command."""

    parser = argparse.ArgumentParser(
        description="Hybrid Search RAG CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable detailed INFO logging.")

    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    # Recommend Command Parser
    parser_recommend = subparsers.add_parser('recommend', help='Get recommendations or answers using RAG.')
    parser_recommend.add_argument('-q', '--query', type=str, default=config.DEFAULT_QUERY, help=f"Query string (default: '{config.DEFAULT_QUERY}')")
    parser_recommend.add_argument('-n', '--num_results', type=int, default=config.TOP_N_RESULTS, help=f'Number of fallback results (default: {config.TOP_N_RESULTS})')
    parser_recommend.add_argument('--general', action='store_true', help='Use Hybrid RAG mode (LLM general knowledge allowed).')
    parser_recommend.add_argument('--concise', action='store_true', help='Use concise/structured prompt template.')

    # Fetch Command Parser
    parser_fetch = subparsers.add_parser('fetch', help='Fetch/process fresh data.')
    parser_fetch.add_argument('--suggest-sources', action='store_true', help="Use LLM to suggest sources based on topic.")
    parser_fetch.add_argument('-t', '--topic', type=str, help="Topic for LLM source suggestion.")
    parser_fetch.add_argument('-aq', '--arxiv_query', type=str, help="Specify custom arXiv query (overrides default).")
    # Ensure MAX_ARXIV_RESULTS exists in config before using it as default
    default_max_arxiv_cli = 10
    if hasattr(config, 'MAX_ARXIV_RESULTS'):
        default_max_arxiv_cli = config.MAX_ARXIV_RESULTS
    parser_fetch.add_argument('-na', '--num_arxiv', type=int, default=default_max_arxiv_cli, help=f"Max arXiv results (default: {default_max_arxiv_cli})")


    # Find Arxiv Command Parser
    parser_find = subparsers.add_parser('find_arxiv', help='Search arXiv directly.')
    parser_find.add_argument('-q', '--query', type=str, required=True, help="Query string for arXiv.")
    parser_find.add_argument('-n', '--num_results', type=int, default=10, help='Max arXiv results (default: 10)')

    args = parser.parse_args()

    # Configure Logging Level based on verbose flag
    log_level = logging.INFO if args.verbose else logging.WARNING
    # Reconfigure logger for CLI specifically
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)-7s - [CLI] %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True)
    logger.info(f"CLI logging level set to: {logging.getLevelName(log_level)}")


    # --- Execute Selected Command ---
    # Moved NLTK check inside commands that need it
    try:
        if args.command == 'recommend':
            check_nltk_data() # Needed for potential tokenization in recommender
            # Note: raw_chunks is still returned but won't be used for highlighting here
            response_gen, sources, _ = run_recommendation(
                args.query, args.num_results, args.general, args.concise
            )
            # Format and print CLI output
            mode_name = "Hybrid" if args.general else "Strict"
            print(f"\n--- Running in {mode_name} RAG Mode ---")
            print(f"Query: '{args.query}'")
            print("\n--- Assistant Response ---")

            full_response_cli = ""
            try:
                print("  ", end="", flush=True) # Initial indent
                for chunk in response_gen:
                    # Replace newlines within the chunk to maintain indent on print
                    print(chunk.replace('\n', '\n  '), end="", flush=True)
                    full_response_cli += chunk
                print() # Final newline after streaming finishes
            except Exception as stream_e:
                print(f"\n[Error during streaming output: {stream_e}]")

            print("\n--- Context Sources Provided to LLM ---")
            print("\n".join(sources) if sources else "  None")
            print("-" * 40)

        elif args.command == 'fetch':
            # NLTK check is now inside setup_data_and_fetch
            status_message = setup_data_and_fetch(args)
            print(f"\n--- Fetch Status ---\n  {status_message}\n" + "-" * 40)

        elif args.command == 'find_arxiv':
            # NLTK check not strictly needed here
            results = run_arxiv_search(args.query, args.num_results)
            print(f"\n--- arXiv Search Results for: '{args.query}' (Top {args.num_results}) ---")
            if not results: print("  No results found.")
            else:
                for i, paper in enumerate(results):
                    print(f"\n{i + 1}. Title: {paper.get('title', 'N/A')}")
                    print(f"   Authors: {', '.join(paper.get('authors', []) or ['N/A'])}")
                    print(f"   Published: {paper.get('published', 'N/A')}")
                    print(f"   Link: {paper.get('pdf_url', '#')}")
                    summary = paper.get('summary', 'N/A')
                    print(f"   Summary: {summary[:400]}{'...' if len(summary) > 400 else ''}")
                    print("-" * 20)
            print("-" * 40)

    except Exception as e:
         logger.critical(f"CLI command execution failed: {e}", exc_info=True)
         print(f"\nERROR: An unexpected error occurred: {e}", file=sys.stderr)
         sys.exit(1)

# Script entry point guard
if __name__ == "__main__":
    main()
