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
# from turtle import st # Incorrect import, remove
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import arxiv
import nltk # Import nltk for data check
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
import textwrap
import traceback # Keep traceback import if used elsewhere, otherwise optional

# --- REMOVE NLTK Download/Path Append Blocks from here ---
# The Streamlit download block using st.* should NOT be here.
# The nltk.data.path.append block for bundling should also be removed
# if using the runtime download approach in app.py.
# ---

# --- Define logger at Module Level ---
logger = logging.getLogger(__name__)

# --- Path Setup ---
# Add project root to sys.path to allow absolute imports from hybrid_search_rag package
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---


# --- Import Project Modules (Order matters sometimes) ---
try:
    from hybrid_search_rag import config
    from hybrid_search_rag.data.resource_fetcher import fetch_arxiv_papers, crawl_and_fetch_web_articles
    # Ensure this points to the correct embedding model
    from hybrid_search_rag.retrieval.embedding_model_gemini import EmbeddingModel
    from hybrid_search_rag.data.data_manager import DataManager
    # Ensure this points to the correct recommender
    from hybrid_search_rag.retrieval.recommender import Recommender, tokenize_text
    from rank_bm25 import BM25Okapi
    from hybrid_search_rag.llm.llm_interface import get_llm_response
except ImportError as e:
    # Logger might not be configured yet if basicConfig fails below,
    # so use print for critical import errors
    print(f"CRITICAL: Failed to import project modules: {e}", file=sys.stderr)
    print("CRITICAL: Ensure you are running this script relative to the project root", file=sys.stderr)
    print(f"CRITICAL: Current sys.path includes: {sys.path}", file=sys.stderr)
    sys.exit(1)
# --- End Import Project Modules ---


# --- NLTK Data Check (for CLI execution) ---
# This function is for when running from the command line
def check_nltk_data():
    """Checks required NLTK data. Exits if missing after download attempt."""
    required_data = {'punkt': 'tokenizers/punkt', 'stopwords': 'corpora/stopwords'}
    missing_data = []
    for name, path in required_data.items():
        try:
            nltk.data.find(path)
        except LookupError:
            missing_data.append(name)

    if missing_data:
        # Attempt download only if running as CLI script
        logger.warning(f"NLTK data packages missing: {', '.join(missing_data)}.")
        print(f"\nAttempting to download missing NLTK data: {', '.join(missing_data)}...", file=sys.stderr)
        try:
            # Add SSL context bypass for CLI download attempt as well
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context

            for name in missing_data:
                print(f"Downloading NLTK package: {name}") # More verbose for CLI
                if nltk.download(name, quiet=False): # Show download progress
                    logger.info(f"Successfully downloaded NLTK data '{name}'.")
                    # Re-verify after download
                    try:
                         nltk.data.find(required_data[name])
                    except LookupError:
                         logger.error(f"Verification failed after downloading '{name}'. Exiting.")
                         sys.exit(1) # Exit CLI if download fails verification
                else:
                    logger.error(f"Failed to download NLTK data '{name}'. Exiting.")
                    sys.exit(1) # Exit CLI if download fails
        except Exception as e:
            logger.error(f"An error occurred during NLTK download: {e}", exc_info=True)
            print("\nAutomatic NLTK download failed. Please try manually in a Python interpreter:", file=sys.stderr)
            print(">>> import nltk", file=sys.stderr)
            for name in missing_data: print(f">>> nltk.download('{name}')", file=sys.stderr)
            sys.exit(1) # Exit CLI if download fails
    else:
        # Only log if running verbosely perhaps, or just skip
        # logger.info("Required NLTK data packages found.")
        pass # Data found, no action needed for CLI

# --- Chunking Helper Function Definition ---
# (chunk_text_by_sentences function remains the same)
def chunk_text_by_sentences(text: str, sentences_per_chunk: int = 5, overlap_sentences: int = 1) -> List[str]:
    """Splits text into chunks based on sentences with overlap."""
    if not text:
        return []

    try:
        # Ensure NLTK punkt tokenizer is available (checked globally by check_nltk_data)
        sentences = nltk.sent_tokenize(text)
    except Exception as e:
        # Use the script's logger instance
        logger.warning(f"Sentence tokenization failed: {e}. Falling back to basic newline split.")
        # Basic fallback: split by paragraph or line, less ideal
        sentences = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not sentences:
             sentences = [p.strip() for p in text.split('\n') if p.strip()]

    if not sentences:
        logger.warning("Could not extract sentences or lines for chunking.")
        return []

    chunks = []
    start_index = 0
    while start_index < len(sentences):
        end_index = min(start_index + sentences_per_chunk, len(sentences))
        chunk_sentences = sentences[start_index:end_index]
        chunks.append(" ".join(chunk_sentences)) # Join sentences in chunk with space

        # Move start index for next chunk, considering overlap
        step = sentences_per_chunk - overlap_sentences
        if step <= 0: # Ensure progression even with large overlap
            step = 1
        start_index += step

    return chunks
# --- END Chunking Helper Function Definition ---


# --- Core Application Logic ---
# (load_components function remains the same)
loaded_metadata: Optional[List[Dict[str, Any]]] = None
loaded_embeddings: Optional[np.ndarray] = None
loaded_bm25_index: Optional[BM25Okapi] = None
loaded_recommender: Optional[Recommender] = None

def load_components(force_reload: bool = False):
    """Loads data and initializes recommender, caching them globally."""
    global loaded_metadata, loaded_embeddings, loaded_bm25_index, loaded_recommender

    if force_reload or loaded_metadata is None or loaded_recommender is None:
        logger.info(f"{'Forcing reload' if force_reload else 'Loading components'}...")
        data_manager = DataManager(config.DATA_DIR, config.METADATA_FILE, config.EMBEDDINGS_FILE, config.BM25_INDEX_FILE)
        loaded_metadata, loaded_embeddings, loaded_bm25_index = data_manager.load_all_data()

        if loaded_metadata is None:
            logger.error("Metadata failed to load. Cannot proceed.")
            raise RuntimeError("Metadata loading failed.")

        try:
            # Ensure EmbeddingModel uses the correct model name from config if needed
            embedder = EmbeddingModel(config.EMBEDDING_MODEL_NAME)
            loaded_recommender = Recommender(embed_model=embedder)
            logger.info("Recommender initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Recommender: {e}", exc_info=True)
            loaded_recommender = None
            raise RuntimeError("Recommender initialization failed.") from e

        # Log status of optional components
        if loaded_embeddings is None: logger.warning("Embeddings missing; semantic search disabled.")
        if loaded_bm25_index is None: logger.warning("BM25 index missing; keyword search disabled.")
        if loaded_embeddings is None and loaded_bm25_index is None:
             logger.error("Both embeddings and BM25 index are missing. Search impossible.")

# (setup_data_and_fetch function remains the same - ensure removed line stays removed)
def setup_data_and_fetch(args: argparse.Namespace) -> str:
    """
    Handles the 'fetch' command logic, optionally using LLM suggestions.
    Returns:
        str: A status message indicating success or failure and key counts.
    """
    logger.info("Starting forced data fetch and processing...")
    status_messages = [] # To collect warnings during the process

    # --- Determine sources based on args ---
    arxiv_query = ""; target_urls = []; max_results = args.num_arxiv
    try:
        if args.suggest_sources:
            logger.info(f"Attempting LLM source suggestion for topic: '{args.topic}'")
            if not args.topic:
                return "Error: Topic (-t) is required when using --suggest-sources." # Return error string


            # Craft prompt for LLM - **NEEDS CAREFUL TUNING**
            # Added profile context and refined instructions for formatting
            suggestion_prompt = f"""You are an expert research assistant helping a machine learning student.
    Based on the topic "{args.topic}", suggest the following:
    1.  Two diverse and effective arXiv search query strings suitable for finding key papers. Format each query on a new line prefixed *exactly* with "ARXIV_QUERY: ".
    2.  Five specific, high-quality, publicly accessible URLs relevant to the topic. Prioritize official conference proceedings (like NeurIPS, ICML, PMLR), key research lab blogs (like OpenAI, Google AI), or highly reputable technical blogs (like distill.pub, lilianweng). Format each URL on a new line prefixed *exactly* with "URL: ".

    Provide *only* the queries and URLs in the specified format, without any introduction, explanation, numbering of sections, or other commentary."""

            logger.info("Sending request to LLM for source suggestions...")
            try:
                suggestion_response = get_llm_response(suggestion_prompt)
                if suggestion_response:
                    logger.info("Received suggestions from LLM. Parsing...")
                    suggested_arxiv_queries = []
                    suggested_urls = []
                    for line in suggestion_response.splitlines():
                        line = line.strip()
                        if line.startswith("ARXIV_QUERY:"):
                            query_term = line.replace("ARXIV_QUERY:", "").strip()
                            if query_term: suggested_arxiv_queries.append(query_term)
                        elif line.startswith("URL:"):
                            url_term = line.replace("URL:", "").strip()
                            if url_term: suggested_urls.append(url_term)

                    if suggested_arxiv_queries:
                        arxiv_query = suggested_arxiv_queries[0]
                        logger.info(f"Using LLM-suggested arXiv query: '{arxiv_query}'")
                    else:
                        logger.warning("LLM did not suggest arXiv queries."); arxiv_query = ""; max_results = 0
                    if suggested_urls:
                        target_urls = suggested_urls
                        logger.info(f"Using {len(target_urls)} LLM-suggested URLs.")
                    else:
                        logger.warning("LLM did not suggest URLs."); target_urls = []
                else:
                     # This part handles LLM call failure or empty response
                     logger.error("LLM failed to provide source suggestions. Clearing sources.")
                     arxiv_query = ""; max_results = 0; target_urls = []
                     status_messages.append("Warning: LLM suggestion failed.") # Add warning

            except Exception as llm_e: # Catch errors during the LLM call itself
                 logger.error(f"Error during LLM suggestion API call: {llm_e}. Clearing sources.", exc_info=True)
                 arxiv_query = ""; max_results = 0; target_urls = []
                 status_messages.append(f"Warning: LLM suggestion failed ({llm_e}).") # Add warning

        elif args.arxiv_query:
            arxiv_query = args.arxiv_query
            target_urls = config.TARGET_WEB_URLS
            logger.info(f"Using user-provided arXiv query: '{arxiv_query}'")
            logger.info("Using default static target URLs from config.")

        else:
            arxiv_query = config.DEFAULT_ARXIV_QUERY
            target_urls = config.TARGET_WEB_URLS
            logger.info("Using default arXiv query and static target URLs from config.")

    except Exception as e:
         logger.error(f"Error during source determination/LLM suggestion: {e}", exc_info=True)
         return f"Error determining sources: {e}" # Return error string

     # --- Log final sources ---
    logger.info(f"Proceeding with arXiv query: '{arxiv_query}' (Max Results: {max_results})")
    logger.info(f"Proceeding with Target URLs: {target_urls[:3]}..." if len(target_urls) > 3 else target_urls)

    # --- Fetching ---
    arxiv_metadata_list = []; web_metadata_list = []
    try:
        if arxiv_query and max_results > 0:
            arxiv_metadata_list = fetch_arxiv_papers(arxiv_query, max_results)
        else: logger.info("Skipping arXiv fetch.")
    except Exception as e: logger.error(f"Failed to fetch arXiv papers: {e}", exc_info=True)
    try:
        if target_urls:
            web_metadata_list = crawl_and_fetch_web_articles(target_urls) # Use the crawler
        else: logger.info("Skipping web/PDF fetch.")
    except Exception as e: logger.error(f"Failed to fetch web articles: {e}", exc_info=True)

    original_documents = (arxiv_metadata_list or []) + (web_metadata_list or [])
    num_docs_fetched = len(original_documents)
    if num_docs_fetched == 0:
        msg = "No data fetched from any source. Aborting processing."
        logger.error(msg)
        return f"Error: {msg}" # Return error string
    logger.info(f"Total original documents fetched: {num_docs_fetched}")

    # --- Chunking ---
    all_chunk_metadata = []
    logger.info("Chunking documents...")
    # ... (Keep your existing chunking loop - uses logger.warning) ...
    for doc_index, doc_meta in enumerate(original_documents):
        original_content = doc_meta.get('content', '') or ''
        original_url = doc_meta.get('url', f'doc_{doc_index}')
        if not original_content.strip(): continue
        # Make sure chunk_text_by_sentences uses the logger defined in this script
        text_chunks = chunk_text_by_sentences(original_content)
        if not text_chunks: continue
        for chunk_index, chunk_text in enumerate(text_chunks):
             chunk_id = f"{original_url}_chunk_{chunk_index}"
             chunk_metadata = {
                 "chunk_id": chunk_id, "chunk_text": chunk_text, "original_url": original_url,
                 "original_title": doc_meta.get('title', 'Untitled Document'), "source": doc_meta.get('source', 'unknown'),
                 "authors": doc_meta.get('authors', []), "published": doc_meta.get('published', None),
                 "arxiv_entry_id": doc_meta.get('entry_id') if doc_meta.get('source') == 'arxiv' else None
             }
             all_chunk_metadata.append(chunk_metadata)

    num_chunks = len(all_chunk_metadata)
    if num_chunks == 0:
        msg = "No chunks were generated from any document. Aborting."
        logger.error(msg)
        return f"Error: {msg}" # Return error string
    logger.info(f"Generated {num_chunks} chunks from {num_docs_fetched} original documents.")

    # --- Processing & Saving ---
    manager = DataManager(config.DATA_DIR, config.METADATA_FILE, config.EMBEDDINGS_FILE, config.BM25_INDEX_FILE)
    # Ensure EmbeddingModel uses the correct model name from config if needed
    embedder = EmbeddingModel(config.EMBEDDING_MODEL_NAME)
    all_chunk_texts = [chunk.get('chunk_text', '') for chunk in all_chunk_metadata]
    embeddings = None; bm25_index = None; embed_success = False; bm25_success = False

    # Embedding
    try:
        # The line 'if embedder.model is None: embedder.load_model()' was removed previously. Keep it removed.
        # Default task_type in encode is RETRIEVAL_DOCUMENT, which is correct here.
        embeddings = embedder.encode(all_chunk_texts)
        if embeddings is None or not isinstance(embeddings, np.ndarray) or embeddings.shape[0] != num_chunks:
             logger.error("Embedding generation failed or returned incorrect shape/None.")
             embeddings = None # Ensure it's None
             # Optionally raise error if embeddings are critical for fetch to succeed
             # raise ValueError("Embedding generation failed or returned incorrect shape.")
        else:
             logger.info(f"Chunk embeddings generated with shape: {embeddings.shape}")
             embed_success = True
    except Exception as e:
        logger.error(f"Failed to generate chunk embeddings: {e}. Semantic search will be unavailable.", exc_info=True)
        status_messages.append("Warning: Embedding generation failed.")
        embeddings = None # Ensure embeddings is None on error

    # BM25 Indexing
    try:
        logger.info("Building BM25 index on chunks...")
        valid_content = [t for t in all_chunk_texts if isinstance(t, str) and t.strip()]
        if valid_content:
            # Ensure tokenize_text uses the logger defined in this script
            tokenized_corpus = [tokenize_text(t) for t in valid_content]
            tokenized_corpus = [tok for tok in tokenized_corpus if tok] # Remove empty lists
            if tokenized_corpus:
                bm25_index = BM25Okapi(tokenized_corpus)
                logger.info("BM25 index built on chunks.")
                bm25_success = True
            else: logger.warning("Tokenized chunk corpus empty after filtering. BM25 index not built.")
        else: logger.warning("No valid chunk text content for BM25 index.")
    except Exception as e:
        logger.error(f"Error building BM25 index on chunks: {e}. Keyword search will be unavailable.", exc_info=True)
        status_messages.append("Warning: BM25 index creation failed.")

    # --- Saving ---
    try:
        manager.save_all_data(all_chunk_metadata, embeddings, bm25_index)
        logger.info("New chunked data fetched, processed, and saved successfully.")
        # Reload components only if save was successful
        load_components(force_reload=True) # Reload components in global scope

        # Construct success message
        success_msg = f"Success: Fetched {num_docs_fetched} documents, generated {num_chunks} chunks. Saved metadata"
        if embed_success: success_msg += ", embeddings"
        if bm25_success: success_msg += ", BM25 index"
        success_msg += "."
        if status_messages: # Append any warnings
            success_msg += " " + " ".join(status_messages)
        return success_msg # Return success string

    except Exception as e:
        logger.error(f"Failed to save processed chunked data after fetch: {e}", exc_info=True)
        # Return error message string
        return f"Error: Failed to save processed data. Check logs. Details: {e}"

# (format_fallback_results function remains the same)
def format_fallback_results(results: List[Tuple[Dict[str, Any], float]], num_to_show: int) -> str:
    """Helper to format fallback retrieval results into a string."""
    if not results:
        return "LLM response failed and no relevant document chunks were found matching the query."

    actual_num_to_show = min(num_to_show, len(results))
    lines = [f"LLM response failed. Showing top {actual_num_to_show} retrieved document chunks (fallback):"]
    displayed_count = 0
    for rank, (chunk_metadata, fused_score) in enumerate(results):
        if displayed_count >= actual_num_to_show: break
        lines.append("-" * 20)
        lines.append(f"{rank + 1}. [Score: {fused_score:.4f}] [Source: {chunk_metadata.get('source', 'N/A')}]")
        lines.append(f"   Title: {chunk_metadata.get('original_title', 'N/A')}")
        lines.append(f"   URL: {chunk_metadata.get('original_url', '#')}")
        lines.append(f"   Snippet: {(chunk_metadata.get('chunk_text', '') or '')[:200]}...")
        displayed_count += 1
    return "\n".join(lines)

# (run_recommendation function remains the same)
def run_recommendation(query: str, num_final_results: int, general_mode: bool, concise_mode: bool) -> Tuple[Optional[str], List[str]]:
    """
    Handles recommendation logic. Returns response string and context source strings.
    """
    mode_name = "Hybrid-Knowledge RAG (Local Docs + General)" if general_mode else "Strict RAG (Local Docs Only)"
    style_name = "Concise/Structured" if concise_mode else "Default/Detailed" # Log the style
    logger.info(f"Running recommendation in {mode_name} Mode ({style_name} Prompt) for query: '{query}'")

    llm_answer_final: Optional[str] = None
    source_info_list: List[str] = []
    hybrid_results: List[Tuple[Dict[str, Any], float]] = []

    try:
        # --- Ensure Components Loaded ---
        if loaded_recommender is None or loaded_metadata is None:
            logger.info("Components not loaded, calling load_components().")
            load_components()
        if loaded_recommender is None or loaded_metadata is None:
             raise RuntimeError("Core components (Recommender/Metadata) could not be loaded. Ensure data exists.")

        # --- Retrieval Step ---
        logger.info("Retrieving top documents from local data for context...")
        num_candidates = max(config.RAG_NUM_DOCS + 5, num_final_results + 5)

        if loaded_metadata:
            num_candidates = min(num_candidates, len(loaded_metadata))
        else:
            num_candidates = 0

        if num_candidates > 0 and loaded_metadata:
             # Ensure recommender uses the logger defined in this script if needed internally
             hybrid_results = loaded_recommender.recommend(
                 query=query,
                 resource_metadata=loaded_metadata,
                 resource_embeddings=loaded_embeddings,
                 bm25_index=loaded_bm25_index,
                 semantic_candidates=config.SEMANTIC_CANDIDATES,
                 keyword_candidates=config.KEYWORD_CANDIDATES,
                 fusion_k=config.RANK_FUSION_K,
                 top_n_final=num_candidates
             )
        else:
            logger.warning("Skipping recommendation step due to missing metadata or zero candidates.")
            hybrid_results = []

        # --- Prepare Context & Sources ---
        context_string = "No relevant document chunks were found in the local data."
        if hybrid_results:
            top_chunks_for_rag = hybrid_results[:config.RAG_NUM_DOCS]
            context_texts = []
            reference_map = {}
            for i, (chunk_meta, score) in enumerate(top_chunks_for_rag):
                chunk_num = i + 1
                title = chunk_meta.get('original_title', 'N/A')
                url = chunk_meta.get('original_url', '#')
                snippet = (chunk_meta.get('chunk_text', '') or '')[:config.MAX_CONTEXT_LENGTH_PER_DOC]
                context_texts.append(f"[{chunk_num}] Title: {title}\nURL: {url}\nSnippet: {snippet}")
                reference_map[chunk_num] = f"{title} (URL: {url})"

            context_string = "\n\n---\n\n".join(context_texts)
            source_info_list = [f"  [{num}] {details}" for num, details in reference_map.items()]
            logger.info(f"Prepared context from {len(top_chunks_for_rag)} retrieved document chunks.")
        else:
            logger.info("No relevant document chunks found for context.")

        # --- Prompt Selection and LLM Call ---
        final_prompt: str = ""
        prompt_template_base = """[INST] {system_message}

Provided Document Chunks:
{context}

User Question: {query} [/INST]
Answer:"""

        # --- Conditional Logic for Prompts ---
        # (This whole section remains the same)
        if general_mode:
            if concise_mode:
                logging.info("Using Hybrid-Knowledge RAG prompt (Concise/Structured)")
                system_message = (
                    "You are a helpful AI assistant.\n"
                    "Provide a structured summary answering the user's question.\n"
                    "Use the provided document chunk excerpts for context and specific details where relevant, but also use your general knowledge for comprehensiveness.\n"
                    "Format your response exactly as follows, using Markdown headers and ensuring line breaks between sections:\n\n"
                    "## What is it about:\n"
                    "[Your 2-7 sentence introduction here]\n\n"
                    "## Key Findings:\n"
                    "[Your 4-10 bullet points summarizing findings here, like:\n* Point 1 [cite]\n* Point 2 [cite]]\n\n"
                    "## Conclusion:\n"
                    "[Your 2-5 sentence conclusion here, mentioning limitations if needed]\n\n"
                    "Cite relevant chunk numbers using bracketed numerals like [1], [2] when using information primarily from the chunks.\n"
                    "At the very end, provide a numbered reference list mapping citations [N] to the source title and URL.\n"
                    "Finally, after the reference list, add a section titled '## Further Reading Suggestions:' and suggest 1-4 specific resources."
                )
            else:
                logging.info("Using Hybrid-Knowledge RAG prompt (Default/Detailed)")
                system_message = (
                     "You are an AI assistant specialized in analyzing technical documents.\n"
                     "Provide a **thorough, detailed, and comprehensive** answer to the user's question based **only** on the provided document chunk excerpts below.\n"
                     "The goal is a well-explained, **multi-paragraph response**. Format your response clearly using the following structure:\n\n"
                     "1.  **Overview Paragraph:** Start with a paragraph providing a general introduction to the main concepts based on the context.\n"
                     "2.  **Detailed Sections:** For each major aspect or topic relevant to the question found in the text:\n"
                     "    * Use Markdown headings (`## Heading Name`) to create distinct sections for each topic.\n"
                     "    * Provide **in-depth explanations** under each heading. **Synthesize information *across multiple relevant document chunks*** where applicable.\n"
                     "    * Aim to develop each key point with approximately **3-5 sentences** of detailed explanation, drawing directly from the provided text.\n"
                     "    * Where possible, **explain the relationships or connections** between different concepts mentioned.\n"
                     "    * Include specific examples or details mentioned in the text.\n"
                     "    * If the text presents different perspectives or nuances, try to **reflect those accurately**.\n"
                     "3.  **Conclusion Paragraph:** End with a paragraph summarizing the main points discussed and clearly stating any limitations based *only* on the provided text excerpts (e.g., if the answer is incomplete due to limited context).\n\n"
                     "**Crucially, base your entire answer ONLY on the provided document chunks.** Do not use any prior knowledge.\n"
                     "When referencing information from a specific chunk, cite its corresponding number using bracketed numerals like `[1]`, `[2]`, etc.\n"
                     "At the very end of your response, provide a numbered reference list mapping each citation number `[N]` to the corresponding source document's original title and URL."
                )
        else: # Strict RAG Mode
            if concise_mode:
                logging.info("Using Strict RAG prompt (Concise/Structured)")
                system_message = (
                     "You are an AI assistant specialized in analyzing technical documents.\n"
                     "Provide a concise summary answering the user's question based only on the provided document chunk excerpts below.\n"
                     "Format your response exactly as follows, using Markdown headers and ensuring line breaks between sections:\n\n"
                     "## What is it about:\n"
                     "[Your 2-3 sentence introduction here]\n\n"
                     "## Key Findings:\n"
                     "[Your 4-8 bullet points summarizing findings here, like:\n* Point 1 [cite]\n* Point 2 [cite]]\n\n"
                     "## Conclusion:\n"
                     "[Your 3-4 sentence conclusion here, mentioning limitations if needed]\n\n"
                     "Cite relevant chunk numbers using bracketed numerals like [1], [2].\n"
                     "At the very end, provide a numbered reference list mapping citations [N] to the source title and URL.\n"
                     "Do not use any prior knowledge."
                )
            else:
                logging.info("Using Strict RAG prompt (Default/Detailed)")
                system_message = (
                     "You are an AI assistant specialized in analyzing technical documents.\n"
                     "Provide a detailed and comprehensive answer to the user's question based *only* on the provided document chunk excerpts below.\n"
                     "Format your response in a clear structure with these elements:\n\n"
                     "1. Start with a brief overview paragraph introducing the main concepts.\n"
                     "2. For each major aspect of the answer:\n"
                     "   - Use headings (## Heading) to separate distinct topics\n"
                     "   - Include detailed explanations with examples where available\n"
                     "   - Develop each key point with 3-5 sentences of explanation\n"
                     "3. End with a conclusion summarizing the key findings\n\n"
                     "Synthesize information across chunks when relevant. When referencing information from a specific chunk, cite its corresponding number using bracketed numerals like [1], [2], etc.\n"
                     "At the very end of your response, provide a numbered reference list mapping each citation number [N] to the corresponding source document's original title and URL (e.g., '[1] Title One - http://example.com/one').\n"
                     "If the chunks do not contain enough information to answer the question completely and accurately, clearly state the limitations based on the provided text. Do not use any prior knowledge."
                )

        # --- Format Final Prompt ---
        final_prompt = prompt_template_base.format(system_message=system_message, context=context_string, query=query)

        logger.info(f"Generating response using {config.LLM_PROVIDER} API ({config.LLM_MODEL_ID})...")
        llm_answer_raw = get_llm_response(final_prompt) # Call LLM

        if llm_answer_raw:
            llm_answer_final = llm_answer_raw.strip() # Assign the successful response
            logger.info("LLM response generated successfully.")
        else:
            logger.warning("LLM generation failed or returned no answer.")
            llm_answer_final = format_fallback_results(hybrid_results, num_final_results)

    except RuntimeError as e: # Catch specific error from load_components
        logger.error(f"Initialization Error during recommendation: {e}", exc_info=True)
        llm_answer_final = f"Error: {e}. Cannot run recommendation."
    except Exception as e:
        logger.error(f"An unexpected error occurred during recommendation execution: {e}", exc_info=True)
        llm_answer_final = f"An error occurred during recommendation: {e}"

    return llm_answer_final, source_info_list

# (run_arxiv_search function remains the same)
def run_arxiv_search(query: str, num_results: int) -> List[Dict[str, Any]]:
    """
    Performs a direct arXiv search. Returns list of result dicts or empty list on error.
    """
    logger.info(f"Searching arXiv for: '{query}' (Top {num_results} results)")
    results_list: List[Dict[str, Any]] = []
    try:
        client = arxiv.Client(page_size=min(num_results, 100), delay_seconds=1.0, num_retries=3)
        search = arxiv.Search(query=query, max_results=num_results, sort_by=arxiv.SortCriterion.Relevance)
        for result in client.results(search):
            results_list.append({
                "title": result.title,
                "authors": [str(a) for a in result.authors],
                "published": result.published.strftime('%Y-%m-%d') if result.published else "N/A",
                "pdf_url": result.pdf_url or 'N/A',
                "summary": (result.summary or '').replace('  ', ' ').strip()
            })
        if not results_list: logger.info("No results found on arXiv for this query.")
        else: logger.info(f"Found {len(results_list)} results on arXiv.")
    except Exception as e:
        logger.error(f"An error occurred during arXiv search: {e}", exc_info=True)
        results_list = []
    return results_list


# --- Main Execution Guard ---
def main():
    """Parses arguments, configures logging, and runs the appropriate action."""

    # --- Argument Parsing ---
    # (Argument parsing remains the same)
    parser = argparse.ArgumentParser(
        description="Hybrid Study Resource Recommender CLI with RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Enable INFO level logging for detailed output.")
    # ... (rest of parser setup) ...
    subparsers = parser.add_subparsers(dest='command', help='Available commands (recommend, fetch, find_arxiv)', required=True)
    # Recommend Command
    parser_recommend = subparsers.add_parser('recommend', help='Get recommendations or answer questions using RAG modes.')
    parser_recommend.add_argument('--concise', action='store_true', help='Generate a concise, structured summary instead of a detailed answer.')
    parser_recommend.add_argument('-q', '--query', type=str, default=config.DEFAULT_QUERY, help=f"Query string for recommendation (default: '{config.DEFAULT_QUERY}')")
    parser_recommend.add_argument('-n', '--num_results', type=int, default=config.TOP_N_RESULTS, help=f'Number of fallback results to display (default: {config.TOP_N_RESULTS})')
    parser_recommend.add_argument('--general', action='store_true', help='Use Hybrid-Knowledge RAG mode (allows LLM general knowledge).')
    # Fetch Command
    parser_fetch = subparsers.add_parser('fetch', help='Force fetch/process fresh data from sources (optionally using LLM suggestions).')
    parser_fetch.add_argument('--suggest-sources', action='store_true',
                              help="Use LLM to suggest arXiv queries and URLs based on the topic provided via -t/--topic.")
    parser_fetch.add_argument('-t', '--topic', type=str, default=None,
                              help="Topic for LLM source suggestion (required if --suggest-sources is used).")
    parser_fetch.add_argument('-aq', '--arxiv_query', type=str, default=None,
                              help="Specify a custom arXiv query string directly (overrides config default, ignored if --suggest-sources is used).")
    parser_fetch.add_argument('-na', '--num_arxiv', type=int, default=config.MAX_ARXIV_RESULTS,
                              help=f"Max arXiv results override (applies to default, custom, or suggested query) (default: {config.MAX_ARXIV_RESULTS})")
    # Find Arxiv Command
    parser_find = subparsers.add_parser('find_arxiv', help='Search arXiv directly for relevant papers based on a query.')
    parser_find.add_argument('-q', '--query', type=str, required=True, help="Query string for arXiv search.")
    parser_find.add_argument('-n', '--num_results', type=int, default=10, help='Max number of arXiv results to display (default: 10)')

    args = parser.parse_args()

    # --- Configure Logging ---
    log_level = logging.INFO if args.verbose else logging.WARNING
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)-7s - %(name)-15s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # --- Run Pre-checks AFTER Logging is Configured ---
    # Ensure this check runs for CLI execution
    check_nltk_data()

    # --- Argument Validation ---
    if args.command == 'fetch' and args.suggest_sources and not args.topic:
        parser.error("--suggest-sources requires -t/--topic to be specified.")

     # --- Execute Command ---
    # (Command execution logic remains the same)
    try:
        if args.command == 'recommend':
            answer, sources = run_recommendation(args.query, args.num_results, args.general, args.concise)
            # Print CLI Output
            mode_name = "Hybrid-Knowledge RAG (Local Docs + General)" if args.general else "Strict RAG (Local Docs Only)"
            print(f"\n--- Running in {mode_name} Mode ---")
            print(f"--- Query: '{args.query}' ---")
            print("\n--- Assistant Response ---")
            if answer:
                try:
                    terminal_width = os.get_terminal_size().columns
                except OSError:
                    terminal_width = 80
                print(textwrap.fill(answer, width=terminal_width, initial_indent="  ", subsequent_indent="  "))
            else:
                print("  No response or fallback generated.")
            print("\n--- Local Document Chunks Provided as Context ---")
            print("\n".join(sources) if sources else "  No context provided to LLM.")
            print("-" * 30)

        elif args.command == 'fetch':
            status_message = setup_data_and_fetch(args)
            # Print CLI Output
            print(f"\n--- Fetch Status ---")
            print(f"  {status_message}")
            print("-" * 30)

        elif args.command == 'find_arxiv':
            results = run_arxiv_search(args.query, args.num_results)
            # Print CLI Output
            print(f"\n--- Searching arXiv for: '{args.query}' (Top {args.num_results}) ---")
            if not results:
                print("  No results found.")
            else:
                for i, paper in enumerate(results):
                    print(f"\n{i + 1}. Title: {paper.get('title', 'N/A')}")
                    print(f"   Authors: {', '.join(paper.get('authors', []))}")
                    print(f"   Published: {paper.get('published', 'N/A')}")
                    print(f"   PDF Link: {paper.get('pdf_url', '#')}")
                    summary = paper.get('summary', 'N/A')
                    print(f"   Summary: {summary[:500]}{'...' if len(summary) > 500 else ''}")
                    print("-" * 20)
            print("-" * 30)

    except Exception as e:
         logger.critical(f"An error occurred during command execution: {e}", exc_info=True)
         sys.exit(1)

if __name__ == "__main__":
    main()
