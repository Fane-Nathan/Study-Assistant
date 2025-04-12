# scripts/cli.py
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
from typing import Optional, Tuple, List, Dict, Any, Union
import textwrap
import traceback
import asyncio 

# --- Logger Setup ---
logger = logging.getLogger(__name__) # Use 'cli' as logger name

# --- Path Setup ---
# Add project root to sys.path to allow imports from hybrid_search_rag package
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---


# --- Project Module Imports ---
try:
    from hybrid_search_rag import config
    from hybrid_search_rag.data.resource_fetcher import fetch_arxiv_papers, crawl_and_fetch_web_articles
    from hybrid_search_rag.retrieval.embedding_model_gemini import EmbeddingModel
    from hybrid_search_rag.data.data_manager import DataManager
    from hybrid_search_rag.retrieval.recommender import Recommender, NltkManager, RecommendationParams
    from rank_bm25 import BM25Okapi
    from hybrid_search_rag.llm.llm_interface import get_llm_response
except ImportError as e:
    # Use print for critical import errors as logger might not be configured yet
    print(f"ERROR: Failed to import project modules: {e}", file=sys.stderr)
    print(f"Ensure you are running from the project root or the path setup is correct.", file=sys.stderr)
    print(f"Current sys.path: {sys.path}", file=sys.stderr)
    sys.exit(1)
# --- End Project Module Imports ---


# --- NLTK Data Check ---
def check_nltk_data():
    """Checks required NLTK data ('punkt', 'stopwords'). Exits if missing after download attempt."""
    required_data = {'punkt': 'tokenizers/punkt', 'stopwords': 'corpora/stopwords'}
    # missing_data = [name for name, path in required_data.items() if not nltk.data.find(path, quiet=True)]
    missing_data = []

    if missing_data:
        logger.warning(f"Missing NLTK data packages: {', '.join(missing_data)}.")
        # Attempt download only when run as script
        print(f"\nAttempting to download missing NLTK data: {', '.join(missing_data)}...", file=sys.stderr)
        try:
            # Bypass SSL verification if needed for download
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError: pass
            else: ssl._create_default_https_context = _create_unverified_https_context

            for name in missing_data:
                print(f"Downloading NLTK package: {name}")
                if nltk.download(name, quiet=False): # Show download progress for CLI
                    logger.info(f"Successfully downloaded NLTK data '{name}'.")
                    # Verify after download
                    try: nltk.data.find(required_data[name])
                    except LookupError:
                        logger.error(f"Verification failed after downloading '{name}'. Exiting.")
                        sys.exit(1)
                else:
                    logger.error(f"Failed to download NLTK data '{name}'. Exiting.")
                    sys.exit(1)
        except Exception as e:
            logger.error(f"An error occurred during NLTK download: {e}", exc_info=True)
            print("\nAutomatic NLTK download failed. Please try manually:", file=sys.stderr)
            print(">>> import nltk")
            for name in missing_data: print(f">>> nltk.download('{name}')")
            sys.exit(1)
    # else: # No need to log if data is found unless verbose logging is on
    #    logger.info("Required NLTK data packages found.")

# --- Text Chunking ---
def chunk_text_by_sentences(text: str, sentences_per_chunk: int = 5, overlap_sentences: int = 1) -> List[str]:
    """Splits text into chunks by sentences with overlap."""
    if not text: return []
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception as e:
        logger.warning(f"Sentence tokenization failed: {e}. Falling back to newline split.")
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
        step = max(1, sentences_per_chunk - overlap_sentences) # Ensure step is at least 1
        start_index += step
    return chunks
# --- End Text Chunking ---

# --- Global Variables for Loaded Components ---
# Cache loaded components globally within the script's execution context
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
            embedder = EmbeddingModel(config.EMBEDDING_MODEL_NAME) # Assumes config.EMBEDDING_MODEL_NAME is correct
            loaded_recommender = Recommender(embed_model=embedder)
            logger.info("Recommender initialized.")
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

    # Determine Sources (Default, Custom, LLM Suggestion)
    arxiv_query = ""; target_urls = []; max_results = args.num_arxiv
    try:
        if args.suggest_sources:
            logger.info(f"Attempting LLM source suggestion for topic: '{args.topic}'")
            if not args.topic: return "Error: Topic (-t) required for --suggest-sources."

            # Simplified prompt for LLM suggestions
            suggestion_prompt = f"""Suggest search sources for the topic "{args.topic}":
1. Two arXiv query strings. Prefix *exactly* with "ARXIV_QUERY: ".
2. Five high-quality, relevant URLs (e.g., conference proceedings, key labs, tech blogs). Prefix *exactly* with "URL: ".
Provide *only* the queries and URLs in the specified format."""

            logger.info("Requesting source suggestions from LLM...")
            try:
                suggestion_response = get_llm_response(suggestion_prompt)
                if suggestion_response:
                    # Parse LLM response
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

        elif args.arxiv_query: # Custom arXiv query provided
            arxiv_query = args.arxiv_query
            target_urls = config.TARGET_WEB_URLS # Use default web URLs
            logger.info(f"Using custom arXiv query: '{arxiv_query}' and default web URLs.")
        else: # Use defaults from config
            arxiv_query = config.DEFAULT_ARXIV_QUERY
            target_urls = config.TARGET_WEB_URLS
            logger.info("Using default arXiv query and web URLs from config.")
    except Exception as e:
         logger.error(f"Error determining sources: {e}", exc_info=True)
         return f"Error deciding sources: {e}"

    # Fetching Documents
    logger.info(f"Fetching: arXiv query='{arxiv_query}' (max={max_results}), URLs={len(target_urls)}")
    arxiv_metadata_list = fetch_arxiv_papers(arxiv_query, max_results) if (arxiv_query and max_results > 0) else []
    
    if target_urls:
        logger.info("Running asynchronous web fetcher...")
        web_metadata_list = asyncio.run(crawl_and_fetch_web_articles(target_urls))
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
        text_chunks = chunk_text_by_sentences(original_content) # Using default chunk/overlap size
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

    # Processing & Saving Chunks (Embeddings & BM25)
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
        tokenized_corpus = [NltkManager.tokenize_text(t) for t in all_chunk_texts if isinstance(t, str) and t.strip()]
        tokenized_corpus = [tok for tok in tokenized_corpus if tok] # Remove empty lists
        if tokenized_corpus:
            bm25_index = BM25Okapi(tokenized_corpus)
            logger.info("BM25 index built.")
            bm25_success = True
        else: logger.warning("No valid tokens found for BM25 index after processing chunks.")
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
def run_recommendation(query: str, num_final_results: int, general_mode: bool, concise_mode: bool) -> Tuple[Optional[str], List[str]]:
    """Handles recommendation: retrieves chunks, prepares context, calls LLM."""
    mode_name = "Hybrid" if general_mode else "Strict"
    style_name = "Concise" if concise_mode else "Detailed"
    logger.info(f"Running recommendation ({mode_name} RAG, {style_name} Prompt) for query: '{query[:100]}...'")

    llm_answer_final: Optional[str] = None
    source_info_list: List[str] = []
    hybrid_results: List[Tuple[Dict[str, Any], float]] = []

    try:
        # Ensure components are loaded
        if loaded_recommender is None or loaded_metadata is None:
            logger.info("Components not loaded, attempting load...")
            load_components()
        if loaded_recommender is None or loaded_metadata is None:
             raise RuntimeError("Core components failed to load. Run 'fetch' command.")

        # Retrieval Step
        logger.info("Retrieving relevant document chunks...")
        # Request slightly more candidates than needed for context/fallback
        num_candidates = max(config.RAG_NUM_DOCS + 5, num_final_results + 5)
        num_candidates = min(num_candidates, len(loaded_metadata))

        if num_candidates > 0:
            rec_params = RecommendationParams(
                semantic_candidates=config.SEMANTIC_CANDIDATES,
                keyword_candidates=config.KEYWORD_CANDIDATES,
                fusion_k=config.RANK_FUSION_K,
                top_n_final=num_candidates # Use num_candidates here
            )
            
            hybrid_results = loaded_recommender.recommend(
                query=query,
                resource_metadata=loaded_metadata,
                resource_embeddings=loaded_embeddings,
                bm25_index=loaded_bm25_index,
                params=rec_params # Pass the created object
            )
            logger.info(f"Retrieved {len(hybrid_results)} candidate chunks.")
        else: logger.warning("Skipping retrieval (no metadata or zero candidates needed).")

        # Prepare Context & Sources for LLM and display
        context_string = "No relevant document chunks were found in the local data."
        if hybrid_results:
            top_chunks_for_rag = hybrid_results[:config.RAG_NUM_DOCS]
            context_texts = []
            reference_map = {}
            for i, (chunk_meta, _) in enumerate(top_chunks_for_rag):
                chunk_num = i + 1
                title = chunk_meta.get('original_title', 'N/A')
                url = chunk_meta.get('original_url', '#')
                snippet = (chunk_meta.get('chunk_text', '') or '')[:config.MAX_CONTEXT_LENGTH_PER_DOC]
                # Format context for LLM prompt
                context_texts.append(f"[{chunk_num}] Title: {title}\nURL: {url}\nSnippet: {snippet}")
                # Format source for display list (no leading spaces needed here)
                reference_map[chunk_num] = f"{title} (URL: {url})"

            context_string = "\n\n---\n\n".join(context_texts)
            # Format for the list to be returned for display (e.g., in Streamlit)
            source_info_list = [f"[{num}] {details}" for num, details in reference_map.items()]
            logger.info(f"Prepared context from {len(top_chunks_for_rag)} chunks.")
        else: logger.info("No relevant document chunks found to provide as context.")

        # Prompt Selection and LLM Call
        final_prompt: str = ""
        prompt_template_base = """[INST] {system_message}

Provided Document Chunks:
{context}

User Question: {query} [/INST]
Answer:"""

        # Decide which system message to use based on modes
        # (Using the previously corrected prompts with simplified citation instruction)
        if general_mode:
            if concise_mode:
                system_message = ( # Hybrid + Concise
                    "You are a helpful AI assistant.\n"
                    "Give a structured summary answering the user's question.\n"
                    "Use the document chunks I provide for details when needed, but also use your general knowledge.\n"
                    "Format your response *exactly* like this, using Markdown headers:\n\n"
                    "## What is it about:\n[...]\n\n"
                    "## Key Findings:\n[...]\n\n"
                    "## Conclusion:\n[...]\n\n"
                    "Cite relevant chunk numbers like [1], [2].\n"
                    "At the very end, create a section `## Citations:`.\n"
                    "Under that heading, provide a numbered list mapping citations [N] used to the source document.\n"
                    "Format each item by copying the Title and URL associated with that number from the 'Provided Document Chunks' section above (e.g., '[N] Title: Actual Title URL: Actual URL').\n"
                    "After Citations, add `## Further Reading Suggestions:`.\n"
                    "Suggest 1-4 relevant resources as a numbered list: `1. Resource Title: <URL or Description>`"
                )
            else:
                system_message = ( # Hybrid + Detailed
                     "You are an AI assistant expert at analyzing technical documents.\n"
                     "Give a **very detailed and complete** answer to the user's question.\n"
                     "Base your answer mainly on the 'Provided Document Chunks', but add relevant general knowledge for clarity.\n"
                     "Structure your answer clearly:\n1. Overview paragraph.\n2. Detailed Sections (use `## Heading` for topics, explain in detail using 3-5 sentences per point, combine info across chunks).\n3. Conclusion paragraph (summarize, mention limits).\n\n"
                     "Cite chunk numbers like [1], [2].\n"
                     "At the end, create `## Citations:`.\n"
                     "Under it, list citations [N] by copying Title/URL from context (e.g., '[N] Title: Actual Title URL: Actual URL').\n"
                     "After Citations, add `## Further Reading Suggestions:`.\n"
                     "Suggest 1-4 resources as a numbered list: `1. Resource Title: <URL or Description>`"
                )
        else: # Strict Mode
            if concise_mode:
                system_message = ( # Strict + Concise
                     "You are an AI assistant expert at analyzing technical documents.\n"
                     "Give a short summary answering the user's question using *only* the 'Provided Document Chunks'.\n"
                     "Format *exactly* like this:\n\n"
                     "## What is it about:\n[...]\n\n"
                     "## Key Findings:\n[...]\n\n"
                     "## Conclusion:\n[...]\n\n"
                     "Cite chunk numbers like [1], [2].\n"
                     "At the end, create `## Citations:`.\n"
                     "Under it, list citations [N] by copying Title/URL from context (e.g., '[N] Title: Actual Title URL: Actual URL').\n"
                     "If chunks are insufficient, state limitations. Do not use prior knowledge."
                )
            else:
                system_message = ( # Strict + Detailed
                     "You are an AI assistant expert at analyzing technical documents.\n"
                     "Give a detailed answer using *only* the 'Provided Document Chunks'.\n"
                     "Structure:\n1. Overview.\n2. Detailed Sections (`## Heading`, 3-5 sentences/point, combine chunks).\n3. Conclusion.\n\n"
                     "Cite chunk numbers like [1], [2].\n"
                     "At the end, create `## Citations:`.\n"
                     "Under it, list citations [N] by copying Title/URL from context (e.g., '[N] Title: Actual Title URL: Actual URL').\n"
                     "If chunks are insufficient, state limitations. **Do not use any outside knowledge.**"
                )

        final_prompt = prompt_template_base.format(system_message=system_message, context=context_string, query=query)

        logger.info(f"Sending request to LLM ({config.LLM_PROVIDER}, {config.LLM_MODEL_ID})...")
        llm_answer_raw = get_llm_response(final_prompt)

        if llm_answer_raw:
            llm_answer_final = llm_answer_raw.strip()
            logger.info("LLM response received.")
        else:
            logger.warning("LLM generation failed or returned empty response.")
            llm_answer_final = format_fallback_results(hybrid_results, num_final_results)

    except RuntimeError as e:
        logger.error(f"Recommendation failed due to component loading error: {e}", exc_info=False) # No need for full traceback here
        llm_answer_final = f"Error: {e}. Cannot run recommendation."
    except Exception as e:
        logger.error(f"Unexpected error during recommendation: {e}", exc_info=True)
        llm_answer_final = f"An error occurred: {e}"

    return llm_answer_final, source_info_list

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
    parser_fetch.add_argument('-na', '--num_arxiv', type=int, default=config.MAX_ARXIV_RESULTS, help=f"Max arXiv results (default: {config.MAX_ARXIV_RESULTS})")

    # Find Arxiv Command Parser
    parser_find = subparsers.add_parser('find_arxiv', help='Search arXiv directly.')
    parser_find.add_argument('-q', '--query', type=str, required=True, help="Query string for arXiv.")
    parser_find.add_argument('-n', '--num_results', type=int, default=10, help='Max arXiv results (default: 10)')

    args = parser.parse_args()

    # Configure Logging Level
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)-7s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', force=True)

    # Run NLTK Check (Important for tokenization)
    check_nltk_data()

    # Input Validation for Fetch Command
    if args.command == 'fetch' and args.suggest_sources and not args.topic:
        parser.error("--suggest-sources requires -t/--topic.")

     # Execute Selected Command
    try:
        if args.command == 'recommend':
            answer, sources = run_recommendation(args.query, args.num_results, args.general, args.concise)
            # Format and print CLI output
            mode_name = "Hybrid" if args.general else "Strict"
            print(f"\n--- Running in {mode_name} RAG Mode ---")
            print(f"Query: '{args.query}'")
            print("\n--- Assistant Response ---")
            if answer:
                try: terminal_width = os.get_terminal_size().columns
                except OSError: terminal_width = 80
                print(textwrap.fill(answer, width=terminal_width, initial_indent="  ", subsequent_indent="  "))
            else: print("  No response generated.")
            print("\n--- Context Sources Provided to LLM ---")
            print("\n".join(sources) if sources else "  None")
            print("-" * 40)

        elif args.command == 'fetch':
            status_message = setup_data_and_fetch(args)
            print(f"\n--- Fetch Status ---\n  {status_message}\n" + "-" * 40)

        elif args.command == 'find_arxiv':
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
         # Optionally print a simpler error to the user
         print(f"\nERROR: An unexpected error occurred: {e}", file=sys.stderr)
         sys.exit(1)

# Script entry point guard
if __name__ == "__main__":
    main()