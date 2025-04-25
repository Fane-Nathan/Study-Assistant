# -*- coding: utf-8 -*-
"""
Streamlit web application for the Study Assistant (User Provided Version).

Provides an enhanced user interface for:
1. Fetching and processing resources (arXiv, web, potentially others via cli functions).
2. Performing hybrid search and RAG (WITHOUT highlighting) to answer questions based on local data.
3. Directly searching arXiv.
4. Displaying information about the project and how it works.
5. Collecting user feedback via an embedded form.
6. Adding content by crawling a single user-provided URL.
"""

import streamlit as st
# --- Page Configuration MUST be the first Streamlit command ---
st.set_page_config(
    page_title="Research Paper Assistant", # Browser tab title
    page_icon="📚",                      # Browser tab icon
    layout="wide",                      # Use full width of the page
    initial_sidebar_state="auto",       # Sidebar behavior (can be used later)
    menu_items={                        # Custom items in the Streamlit menu (top right)
        'Get Help': 'https://github.com/Fane-Nathan/Study-Assistant', # Link to project repo
        'Report a bug': "https://tally.so/r/n0kkp6",                  # Link to feedback form
        'About': "# About This Project\nThis app helps explore academic research using RAG." # Simple about text
    }
)

# --- Other Imports ---
import logging
import math
import nltk
# import ssl # ssl import seems unused, can be removed if not needed
import os
import sys
import argparse
import traceback
import streamlit.components.v1 as components
# from thefuzz import fuzz # fuzz import removed as highlighting is removed
from typing import List, Dict, Any, Tuple, Generator, Optional # Added Generator, Optional
import asyncio # For async web crawl
import numpy as np # For handling embeddings
import pickle # For BM25 index loading/saving (indirectly via DataManager)
from rank_bm25 import BM25Okapi # For rebuilding BM25 index

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [App] %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# --- Path Setup ---
logger.info("Setting up sys.path...")
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
logger.info(f"Project Root added to sys.path: {project_root}")
# --- End Path Setup ---


# --- Project Module Imports ---
# Highlighting import is REMOVED.
logger.info("Attempting project module imports...")
project_modules_loaded = False
# highlighting_available = False # Flag removed as feature is removed
try:
    from hybrid_search_rag import config
    from scripts.cli import (
        load_components,
        run_recommendation, # Expects (gen, formatted_src, raw_chunks)
        setup_data_and_fetch,
        run_arxiv_search,
        chunk_text_by_sentences, # Import chunking function
        check_nltk_data          # Import NLTK check function
    )
    # Import necessary components for manual data handling
    from hybrid_search_rag.data.data_manager import DataManager
    from hybrid_search_rag.data.resource_fetcher import crawl_and_fetch_web_articles
    from hybrid_search_rag.retrieval.embedding_model_gemini import EmbeddingModel # Assuming Gemini embeddings
    from hybrid_search_rag.retrieval.recommender import NltkManager # For tokenizing BM25

    # REMOVED: Import for highlighting
    # from hybrid_search_rag.utils.highlighting import highlight_sources_fuzzy
    # highlighting_available = True # Flag removed

    project_modules_loaded = True
    logger.info("Project module imports successful (including highlighting).") # Updated log message slightly
except ImportError as e:
    st.error(f"Failed to import project modules (ImportError). Check setup and ensure scripts/cli.py and hybrid_search_rag package are accessible. Error: {e}")
    st.code(f"Current sys.path: {sys.path}")
    logger.error(f"Project import failed (ImportError): {e}", exc_info=True)
except Exception as e:
    # Catch other errors during import
    st.error(f"Unexpected error during project imports: {e}. App functionality will be limited.")
    logger.error(f"Project import failed (Exception): {e}", exc_info=True)
    # If the error is the Prometheus one, add a specific hint
    if 'Duplicated timeseries' in str(e):
        st.warning("Hint: The 'Duplicated timeseries' error often relates to Prometheus metrics. Ensure they are removed or handled correctly in resource_fetcher.py if not needed.")
# --- End Project Module Imports ---

# --- NLTK Data Download Logic ---
# Reuse the NLTK check function from cli.py if modules loaded
if project_modules_loaded:
    try:
        if 'nltk_data_checked_app' not in st.session_state:
             with st.spinner("Checking NLTK data..."):
                 logger.info("Running initial NLTK check via imported function...")
                 check_nltk_data() # Call the function from cli.py
                 st.session_state.nltk_data_checked_app = True
                 logger.info("Initial NLTK check complete.")
    except NameError:
         st.error("NLTK check function `check_nltk_data` not found. Manual NLTK check block needed.")
         pass
    except SystemExit: # Catch SystemExit if NLTK download fails during init
          st.error("Fatal Error: Failed to download required NLTK data during startup. App cannot continue.")
          logger.critical("NLTK download failed during initial check. Stopping app.")
          st.stop() # Stop app execution
    except Exception as nltk_e:
         st.error(f"Error during initial NLTK check: {nltk_e}")
         logger.error(f"NLTK check failed during app init: {nltk_e}", exc_info=True)
         st.stop() # Stop if essential NLTK check fails
else:
    logger.warning("Skipping NLTK check as project modules failed to load.")

# --- Title, Introduction, and Disclaimer (Enhanced) ---
# (Remains unchanged)
st.title("📚 Research Paper Assistant")
st.subheader("Unlock insights from academic literature with AI")
st.markdown("""
Explore academic research effortlessly. This tool leverages Retrieval-Augmented Generation (RAG)
and Hybrid Search to find relevant papers and answer your questions using document context.
""")
with st.expander("ℹ️ Important Disclaimer", expanded=False):
    st.warning("""
    **Please Note:** This tool is specialized for finding and analyzing information within the indexed research papers.
    It excels at providing context-aware answers based on its knowledge base.

    It is **not** a general-purpose chatbot (like ChatGPT, Gemini, etc.) and may not perform well on:
    * General knowledge questions outside the scope of the indexed documents.
    * Creative writing or conversational tasks.
    * Real-time information (e.g., news, weather).
    """, icon="⚠️")
st.divider()
# --- End Title/Intro/Disclaimer ---


# --- Caching Components ---
@st.cache_resource
def cached_load_components():
    """
    Loads core RAG components using the imported 'load_components' function.
    Uses session state to force reload after data updates.
    """
    logger.info("Executing cached RAG component loading logic...")
    if not project_modules_loaded:
        logger.error("Cannot load components: project modules failed import prior to this call.")
        return False, "Core project modules failed to import."

    # --- FIXED: Check session state flag to force reload ---
    force_reload_flag = st.session_state.get("force_component_reload", False)
    if force_reload_flag:
        logger.info("Force reload flag set, calling load_components with force_reload=True.")
        # Clear the flag immediately after reading it
        st.session_state.force_component_reload = False
    # --- End Fix ---

    try:
        # Ensure NLTK is checked before loading components
        check_nltk_data()
        # --- FIXED: Pass the force_reload_flag to load_components ---
        load_components(force_reload=force_reload_flag)
        # --- End Fix ---

        # Check if the recommender object exists in cli.py after loading
        from scripts.cli import loaded_recommender
        if loaded_recommender is not None:
             logger.info("Component loading logic successful (based on 'loaded_recommender' indicator).")
             return True, None
        else:
             logger.warning("load_components ran but indicator 'loaded_recommender' is None. Assuming load failed.")
             return False, "Components appear missing after loading attempt (recommender is None)."
    except NameError as ne:
         logger.error(f"NameError during component loading check: {ne}. Cannot verify loaded state.", exc_info=True)
         return False, f"Error accessing loaded state indicator from 'scripts.cli'. NameError: {ne}"
    except SystemExit: # Catch NLTK download failure during component load
         logger.critical("NLTK download failed during component loading. Cannot proceed.")
         return False, "Failed to acquire NLTK data during component loading."
    except Exception as e:
        logger.error(f"Exception during component loading: {e}", exc_info=True)
        return False, f"An exception occurred during component loading: {e}"
# --- End Caching Components ---


# --- Initial Setup ---
# (Remains unchanged)
logger.info("Attempting initial component load via cached function...")
components_loaded_status, load_error_message = False, "Project modules did not load."
if project_modules_loaded:
    components_loaded_status, load_error_message = cached_load_components()
logger.info(f"Components loaded status: {components_loaded_status}")
if not components_loaded_status and project_modules_loaded:
    st.error(f"Core RAG components failed to load: {load_error_message}. Recommendation and Fetch Data functionality disabled.")


# --- UI Tabs (with Icons) ---
# (Remains unchanged)
logger.info("Defining UI tabs...")
tab_rec, tab_how, tab_about, tab_fetch, tab_arxiv, tab_feedback = st.tabs([
    "🧠 **Recommend**",
    "⚙️ How It Works",
    "ℹ️ About",
    "⏬ Update Knowledge Base",
    "🔍 Search arXiv",
    "📝 Feedback"
])
logger.info("UI tabs defined.")


# --- Recommendation Tab (Highlighting Removed) ---
# (Remains unchanged from user's provided version)
with tab_rec:
    st.header("💬 Ask the RAG Assistant")
    st.markdown("Enter your research topic or question below. The assistant will retrieve relevant information from the indexed documents and generate an answer.") # Removed "cited"

    col1_rec, col2_rec = st.columns([3, 1])

    with col1_rec:
        default_query = config.DEFAULT_QUERY if project_modules_loaded else "Example: Explain Retrieval-Augmented Generation (RAG)."
        query = st.text_area(
            "Your Question:",
            value=st.session_state.get("rec_query", default_query),
            height=150,
            key="rec_query_input_area",
            placeholder="Type your question about the research papers here...",
            help="Ask anything related to the content of the indexed documents.",
            label_visibility="collapsed"
        )

    with col2_rec:
        st.markdown("**Options**")
        general_mode = st.toggle(
            "Hybrid Mode",
            value=st.session_state.get("rec_general", False),
            key="rec_general_toggle",
            help="Allows the AI to use its general knowledge *in addition* to the retrieved documents. 'Strict Mode' (off) uses *only* document context."
        )
        concise_mode = st.toggle(
            "Concise Prompt",
            value=st.session_state.get("rec_concise", True),
            key="rec_concise_toggle",
            help="Uses a more structured, concise prompt for the AI, potentially faster but less conversational."
        )
        st.markdown("---")
        submit_rec = st.button(
            "✨ Get Recommendation",
            type="primary",
            key="rec_button",
            disabled=not components_loaded_status or not query,
            use_container_width=True
        )

    # --- Logic when Button is Clicked ---
    if submit_rec:
        if not components_loaded_status:
            st.error("Cannot recommend: Core components failed load.")
        else:
            st.session_state.rec_query = query
            st.session_state.rec_general = general_mode
            st.session_state.rec_concise = concise_mode
            mode_name = "Hybrid" if general_mode else "Strict"
            style_name = "Concise" if concise_mode else "Detailed"
            logger.info(f"Running recommendation for query: '{query[:50]}...' with Mode='{mode_name}', Style='{style_name}'")
            st.info(f"Running recommendation with '{mode_name}' RAG and '{style_name}' Prompt...", icon="⏳")

            # Clear previous answer and sources before generating new ones
            st.session_state.llm_answer = None
            st.session_state.context_sources = []
            st.session_state.raw_context_chunks = []
            answer_placeholder = st.empty() # Create placeholder for streaming output

            with st.spinner("🧠 Thinking... Retrieving context and generating response..."):
                try:
                    top_n = config.TOP_N_RESULTS if project_modules_loaded else 5
                    # raw_context_chunks is still returned by run_recommendation but ignored here (_)
                    # Call the function from cli.py
                    response_generator, context_sources, _ = run_recommendation(
                        query, top_n, general_mode, concise_mode
                    )
                    st.session_state.context_sources = context_sources # Store sources for expander

                    logger.info("Streaming response...")
                    full_response_list = []
                    error_in_stream = False # Flag to track if error message was yielded
                    for chunk in response_generator:
                        # Check for error messages yielded by the generator
                        if isinstance(chunk, str) and chunk.startswith("[Error:"):
                             st.error(f"LLM Error: {chunk}", icon="❌")
                             logger.error(f"LLM generation failed: {chunk}")
                             # Optionally clear placeholder or stop processing
                             answer_placeholder.empty()
                             st.session_state.llm_answer = None # Ensure no answer is stored
                             error_in_stream = True # Set flag
                             break # Stop processing the stream

                        full_response_list.append(chunk)
                        # Update placeholder with cumulative response for streaming effect
                        # Add a blinking cursor effect (optional)
                        answer_placeholder.markdown("".join(full_response_list) + "▌", unsafe_allow_html=True)

                    # Only process if no error occurred during streaming
                    if not error_in_stream:
                        full_response = "".join(full_response_list)
                        logger.info(f"Full response received (length: {len(full_response)}).")
                        st.session_state.llm_answer = full_response # Store final answer
                        # Display final answer without cursor
                        answer_placeholder.markdown(full_response, unsafe_allow_html=True)
                        st.toast("Response generated!", icon="✅")
                    else:
                        # If an error message was yielded, ensure placeholder is cleared
                        answer_placeholder.empty()


                except Exception as e:
                    st.error(f"Error during recommendation: {e}", icon="❌")
                    logger.error(f"Recommendation Error: {e}", exc_info=True)
                    # Clear results on error
                    st.session_state.llm_answer = None
                    st.session_state.context_sources = []
                    st.session_state.raw_context_chunks = []
                    answer_placeholder.empty() # Clear placeholder on error


    # --- Displaying the Response and Sources (HIGHLIGHTING REMOVED) ---
    # Display final answer *after* streaming is complete or if already in session state
    # This block now only handles displaying an existing answer if the button wasn't just clicked
    elif "llm_answer" in st.session_state and st.session_state.llm_answer is not None:
        # If the button wasn't just clicked (i.e., displaying existing answer from session state)
        st.markdown("---")
        st.subheader("Assistant's Response")
        st.markdown(st.session_state.llm_answer, unsafe_allow_html=True)

    # Display sources expander if sources exist (independent of whether button was clicked)
    if "context_sources" in st.session_state and st.session_state.context_sources:
        # Ensure there's a visual separation if the answer was just streamed
        if submit_rec: st.markdown("---") # Add separator only if we just submitted

        with st.expander("📚 Show Context Sources Provided to LLM", expanded=False):
            sources = st.session_state.context_sources
            st.caption("These are the document snippets retrieved and used to generate the answer above:")
            num_sources = len(sources)
            if num_sources > 0:
                num_columns = 2
                cols = st.columns(num_columns)
                midpoint = math.ceil(num_sources / num_columns)
                with cols[0]:
                    for i in range(midpoint):
                        if i < num_sources: st.markdown(f"{sources[i]}")
                with cols[1]:
                    for i in range(midpoint, num_sources):
                        if i < num_sources: st.markdown(f"{sources[i]}")
# --- End Recommendation Tab ---


# --- How It Works Tab ---
with tab_how:
    st.header("⚙️ How the RAG System Works")
    st.markdown("This system uses a **Retrieval-Augmented Generation (RAG)** pipeline to provide informed answers:")
    col1_how, col2_how = st.columns(2)
    with col1_how:
        st.subheader("1️⃣ Data Ingestion & Processing")
        st.markdown("""
        * **Sources:** Fetches data from arXiv, web URLs, potentially local files.
        * **Cleaning:** Prepares text for analysis.
        * **Chunking:** Breaks down documents into smaller pieces.
        """)
        st.subheader("2️⃣ Hybrid Indexing")
        emb_model = f"`{config.EMBEDDING_MODEL_NAME}`" if project_modules_loaded else "a sentence transformer model"
        st.markdown(f"""
        * **Vector Embeddings:** Creates numerical representations (vectors) capturing semantic meaning using {emb_model}.
        * **Keyword Index (BM25):** Creates a traditional keyword index for term matches.
        * **Combined Power:** Stores both index types locally.
        """)
    with col2_how:
        st.subheader("3️⃣ Hybrid Retrieval (RRF)")
        st.markdown("""
        * **Dual Search:** Your query searches *both* the vector index and the keyword index.
        * **Smart Ranking (RRF):** Results are combined using Reciprocal Rank Fusion (RRF) for a balanced relevance ranking.
        """)
        st.subheader("4️⃣ LLM Generation")
        # --- FIXED: Use new config structure ---
        llm_provider_display = "a Large Language Model (LLM)" # Default
        if project_modules_loaded and config.LLM_PROVIDER_ORDER:
            # Display the primary provider (first in the list)
            llm_provider_display = config.LLM_PROVIDER_ORDER[0].capitalize()
            if len(config.LLM_PROVIDER_ORDER) > 1:
                 # Show fallback only if it's different from primary
                 fallback_provider = config.LLM_PROVIDER_ORDER[1].capitalize()
                 if fallback_provider != llm_provider_display:
                      llm_provider_display += f" (with fallback to {fallback_provider})"
        # --- End Fix ---
        st.markdown(f"""
        * **Context Injection:** Top-ranked retrieved chunks are passed as context to {llm_provider_display}.
        * **Informed Answering:** The LLM generates an answer based on the provided context (Strict RAG) or a mix (Hybrid Mode).
        """)
    st.divider()
    st.subheader("🚀 The RAG Pipeline Advantage")
    st.markdown("""
    * Grounding answers in specific, retrieved information.
    * Reducing the likelihood of hallucinations.
    * Providing transparency through retrieved context (shown in expander).
    """)
    st.subheader("💻 Core Implementation Notes")
    st.markdown("""
    * Built using Python with libraries like `google-generativeai`, `groq`, `rank_bm25`, `nltk`, vector stores, and `Streamlit`.
    * Features modular components for data handling, indexing, retrieval, and generation.
    """)
    st.divider()
# --- End How It Works Tab ---


# --- About Tab ---
# (Remains unchanged)
with tab_about:
    st.header("ℹ️ About This Project")
    col1_about, col2_about = st.columns(2)
    with col1_about:
        st.subheader("🎯 Project Goal")
        st.markdown("""
        To develop an intelligent assistant that helps users efficiently navigate, understand, and discover information within a specific corpus of academic documents.
        """)
        st.subheader("🛠️ Current Status & Features")
        st.markdown("""
        * **RAG Core:** Functional RAG pipeline implemented.
        * **Data Sources:** Ingests arXiv papers, web links.
        * **Hybrid Search:** Combines semantic (vector) and keyword (BM25) search with RRF.
        * **LLM Integration:** Uses Gemini/Groq for generation (Strict/Hybrid modes) with fallback.
        * **UI:** Interactive Streamlit interface for querying, data fetching, and arXiv search.
        """) # Updated LLM integration description
    with col2_about:
        st.subheader("🚀 Future Work & Ideas")
        st.markdown("""
        * **Corpus Expansion:** Curate and index a larger, more diverse set of relevant papers.
        * **Evaluation & Tuning:** Systematically evaluate retrieval and generation quality.
        * **Recommender System:** Proactively suggest relevant papers or topics.
        * **Personalization:** Allow user profiles, history tracking.
        * **Deployment:** Explore options for more robust deployment.
        """)
        st.subheader("👨‍💻 Developers")
        st.markdown(f"""
        Developed by **Felix Nathaniel**, **Reynaldi Anatyo**, & **Dennison Soedibjo**.

        *Computer Science Students at BINUS University, exploring the fascinating world of AI and Information Retrieval.*
        """)
        st.link_button("View Project on GitHub", "https://github.com/Fane-Nathan/Study-Assistant")
# --- End About Tab ---


# --- Fetch Data Tab (With Single URL Crawl Added) ---
with tab_fetch:
    st.header("⏬ Update Knowledge Base")
    st.markdown("""
    Add new documents to the assistant's knowledge base using one of the methods below.
    """)
    st.warning("Fetching and processing new data can take significant time and requires a stable internet connection.", icon="⏳")

    fetch_disabled = not project_modules_loaded
    if fetch_disabled:
        st.error("Data fetching disabled: Core project modules failed to load.", icon="🚫")

    st.subheader("Method 1: Add Content from a Single URL")
    st.markdown("""
    Enter a web page URL. The assistant will fetch its content and any directly linked PDFs, process them, and add them to the knowledge base.
    Please use https://openreview.net/ for research paper, it will be 100% processed and effective.
    """)
    single_url_crawl_input = st.text_input(
        "URL to Fetch:",
        key="single_url_crawl_input",
        placeholder="https://example.com/research_paper_page",
        help="Paste the full URL of the page you want to add.",
        disabled=fetch_disabled
    )
    submit_single_url_crawl = st.button(
        "Fetch and Add Single URL + Linked PDFs", # Updated button text
        key="single_url_crawl_button",
        disabled=fetch_disabled or not single_url_crawl_input
    )

    # --- Single URL Crawl Logic ---
    if submit_single_url_crawl and not fetch_disabled:
        if not (single_url_crawl_input.startswith("http://") or single_url_crawl_input.startswith("https://")):
            st.error("Invalid URL. Please include http:// or https://")
        else:
            with st.spinner(f"Processing URL and linked PDFs: {single_url_crawl_input}... This involves fetching, chunking, embedding, and saving."): # Updated spinner text
                process_completed_without_error = False # Flag to track success
                try:
                    logger.info(f"Starting single URL processing for: {single_url_crawl_input}")

                    # --- Ensure NLTK data is available before processing ---
                    check_nltk_data() # Explicit check before this specific process

                    # --- FIXED: Call crawler without max_pages_override ---
                    # This allows it to process the main page AND linked PDFs up to the config limit
                    crawled_data = asyncio.run(crawl_and_fetch_web_articles(
                        start_urls=[single_url_crawl_input],
                        process_pdfs_linked=True # Ensure PDF processing is enabled
                        # max_pages_override=1 # REMOVED
                    ))
                    # --- End Fix ---

                    if not crawled_data:
                        st.warning(f"Could not fetch or extract content from {single_url_crawl_input} or its linked PDFs.")
                        raise StopIteration("Crawling failed or returned no data.")


                    logger.info(f"Crawled {len(crawled_data)} total items (HTML + PDFs) starting from {single_url_crawl_input}")

                    # --- Process all crawled items (HTML page + PDFs) ---
                    new_chunk_metadata_list = []
                    chunks_to_embed = []
                    processed_urls = set() # Keep track of URLs processed in this batch

                    # Load existing data ONCE before processing crawled items
                    local_data_manager = DataManager(config.DATA_DIR, config.METADATA_FILE, config.EMBEDDINGS_FILE, config.BM25_INDEX_FILE)
                    existing_metadata, existing_embeddings, existing_bm25 = local_data_manager.load_all_data()
                    if existing_metadata is None: existing_metadata = []
                    if existing_embeddings is None:
                         existing_embeddings = np.array([]).reshape(0, config.EMBEDDING_DIM)
                    existing_chunk_texts = {chunk['chunk_text'] for chunk in existing_metadata} if existing_metadata else set()

                    # Check NLTK data again before chunking loop
                    check_nltk_data()

                    for item_index, doc_meta in enumerate(crawled_data):
                        item_url = doc_meta.get('url', f'crawled_item_{item_index}')
                        item_title = doc_meta.get('title', 'Untitled')
                        item_content = doc_meta.get('content', '')
                        item_source_type = doc_meta.get('source', 'unknown') # html or pdf

                        if not item_content:
                             logger.warning(f"Skipping item {item_url} - no content extracted.")
                             continue

                        logger.info(f"Processing item: {item_title} ({item_url})")
                        processed_urls.add(item_url)

                        # Chunk the content of this item
                        try:
                            item_chunks_text = chunk_text_by_sentences(item_content)
                        except Exception as chunk_err:
                             logger.error(f"Failed to chunk item {item_url}: {chunk_err}", exc_info=True)
                             continue # Skip this item if chunking fails

                        if not item_chunks_text:
                            logger.warning(f"No text chunks generated from item: {item_url}")
                            continue

                        # Create metadata for new chunks of this item and check duplicates
                        item_new_chunks_count = 0
                        for chunk_index, chunk_text in enumerate(item_chunks_text):
                            if chunk_text and chunk_text not in existing_chunk_texts:
                                new_meta = {
                                    "chunk_id": f"{item_url}_chunk_{chunk_index}",
                                    "chunk_text": chunk_text,
                                    "original_url": item_url,
                                    "original_title": item_title,
                                    "source": item_source_type, # Use the determined type (html/pdf)
                                    "authors": doc_meta.get('authors', []), # Usually empty for web/pdf
                                    "published": doc_meta.get('published', None), # Usually empty for web/pdf
                                    "arxiv_entry_id": None
                                }
                                new_chunk_metadata_list.append(new_meta)
                                chunks_to_embed.append(chunk_text)
                                existing_chunk_texts.add(chunk_text) # Add to set for intra-batch check
                                item_new_chunks_count += 1
                            elif not chunk_text:
                                logger.debug(f"Skipping empty chunk from {item_url}")

                        if item_new_chunks_count > 0:
                             logger.info(f"Found {item_new_chunks_count} new unique chunks for item: {item_url}")
                        else:
                             logger.warning(f"No new unique chunks found for item: {item_url}")
                    # --- End loop through crawled items ---


                    if not new_chunk_metadata_list:
                        logger.warning(f"No new, unique, non-empty content chunks found for the initial URL ({single_url_crawl_input}) and its linked PDFs after comparing with existing data.")
                        st.warning(f"The content from {single_url_crawl_input} and its linked PDFs did not contain any new information or was already processed.")
                        raise StopIteration("No new unique chunks found overall.")

                    logger.info(f"Generated {len(new_chunk_metadata_list)} total new unique chunks from {len(processed_urls)} processed items.")

                    # Embed ALL new chunks together
                    embedder = EmbeddingModel(config.EMBEDDING_MODEL_NAME)
                    new_embeddings = embedder.encode(chunks_to_embed, task_type="RETRIEVAL_DOCUMENT")

                    if new_embeddings is None or new_embeddings.shape[0] != len(new_chunk_metadata_list):
                         st.error("Embedding failed for the new chunks.")
                         raise StopIteration("Embedding failed.")

                    logger.info(f"Generated {new_embeddings.shape[0]} embeddings for new chunks.")

                    # Combine old and new data
                    combined_metadata = existing_metadata + new_chunk_metadata_list
                    if existing_embeddings.size > 0:
                        combined_embeddings = np.vstack((existing_embeddings, new_embeddings))
                    else:
                        combined_embeddings = new_embeddings

                    logger.info(f"Combined data: {len(combined_metadata)} total chunks.")

                    # Rebuild BM25 index
                    combined_chunk_texts = [chunk['chunk_text'] for chunk in combined_metadata]
                    try:
                        check_nltk_data() # Explicit check right before tokenization
                        tokenized_corpus = [NltkManager.tokenize_text(t) for t in combined_chunk_texts if isinstance(t, str) and t.strip()]
                        tokenized_corpus = [tok for tok in tokenized_corpus if tok] # Remove empty lists
                        if tokenized_corpus:
                            new_bm25_index = BM25Okapi(tokenized_corpus)
                            logger.info("Rebuilt BM25 index for combined corpus.")
                        else:
                            new_bm25_index = None
                            logger.warning("Could not rebuild BM25 index (no valid tokens?).")
                    except SystemExit: # Catch NLTK download failure here too
                         logger.error("NLTK download failed during BM25 rebuild. Skipping index.")
                         st.error("Failed to download NLTK data needed for keyword index. Index not updated.")
                         new_bm25_index = None # Ensure index is None if check fails
                    except Exception as tokenize_err:
                         logger.error(f"Error during tokenization/BM25 rebuild: {tokenize_err}", exc_info=True)
                         st.error(f"Error rebuilding keyword index: {tokenize_err}")
                         new_bm25_index = None # Ensure index is None on error


                    # Save the combined data
                    local_data_manager.save_all_data(combined_metadata, combined_embeddings, new_bm25_index)
                    logger.info("Successfully saved updated data files.")

                    # Reload components
                    st.success(f"Successfully added {len(new_chunk_metadata_list)} new chunks from {len(processed_urls)} items (HTML page + linked PDFs).", icon="✅")
                    st.info("Triggering component reload...")
                    st.session_state.force_component_reload = True
                    process_completed_without_error = True # Mark success

                except StopIteration as stop_e:
                     # Catch the specific StopIteration exceptions we raised for warnings/failures
                     logger.info(f"Stopping processing for {single_url_crawl_input}: {stop_e}")
                     # The warning/error message was already displayed by st.warning/st.error
                     pass # Do nothing else, just let the spinner finish
                except SystemExit: # Catch NLTK download failure from the check at the start of the try block
                     st.error("Fatal Error: Failed to download required NLTK data. Cannot process URL.")
                     logger.critical("NLTK download failed during single URL processing.")
                     # No rerun needed here
                except Exception as e:
                    # Log unexpected errors and show an error message
                    logger.error(f"Error processing single URL {single_url_crawl_input}: {e}", exc_info=True)
                    st.error(f"Failed to add URL: {e}")
                    # No rerun needed here

            # Rerun only if process completed without error
            if process_completed_without_error:
                 st.rerun()


    st.divider() # Separator

    st.subheader("Method 2: Bulk Fetch (arXiv Query or LLM Suggestion)")
    # (Bulk fetch UI and logic remains unchanged)
    st.markdown("""
    Use this method for broader updates:
    * **Use Custom arXiv Query:** Enter keywords to search arXiv. The system will *also* fetch web pages listed in the project's default configuration (`config.TARGET_WEB_URLS`). PDFs linked from these pages will also be processed by default.
    * **Use LLM Suggestion:** Enter a topic, and the assistant will ask the LLM to suggest relevant arXiv queries *and* web URLs to fetch. PDFs linked from suggested URLs will also be processed by default. (NOT RECOMMENDED)
    """) # Updated description slightly

    fetch_options = ("Use arXiv Query", "Use LLM Suggestion (Not Recommended)")
    fetch_method = st.radio(
        "Choose bulk fetch method:",
        options=fetch_options,
        index=0,
        key="bulk_fetch_method_radio",
        horizontal=True,
        disabled=fetch_disabled,
        label_visibility="collapsed"
    )

    st.markdown("---")

    current_bulk_selection = st.session_state.get("bulk_fetch_method_radio", fetch_options[0])
    col1_fetch, col2_fetch = st.columns(2)

    with col1_fetch:
        custom_arxiv_query_value = st.text_input(
            "Custom arXiv Query (for Bulk Fetch):",
            key="bulk_fetch_arxiv_query_val",
            placeholder="e.g., 'attention mechanism transformer'",
            help="Enter keywords for arXiv. Default web URLs from config also fetched.",
            disabled=fetch_disabled or (current_bulk_selection == "Use LLM Suggestion (Requires Topic)")
        )
        topic_value = st.text_input(
            "Topic for LLM Suggestion (for Bulk Fetch):",
            key="bulk_fetch_topic_val",
            placeholder="e.g., 'Latest advancements in RAG'",
            help="Enter topic for LLM to suggest arXiv queries and web URLs.",
            disabled=fetch_disabled or (current_bulk_selection == "Use Custom arXiv Query")
        )

    with col2_fetch:
        default_max_arxiv = 10
        if project_modules_loaded and hasattr(config, 'MAX_ARXIV_RESULTS'):
            default_max_arxiv = config.MAX_ARXIV_RESULTS
        num_arxiv_results_value = st.number_input(
            f"Max arXiv Results (for Bulk Fetch):",
            min_value=5, max_value=500,
            value=st.session_state.get("bulk_fetch_num_results_val", default_max_arxiv),
            step=5, key="bulk_fetch_num_results_val",
            help="Maximum arXiv papers to retrieve for bulk processing.",
            disabled=fetch_disabled
        )

    st.markdown("---")

    submitted_bulk_fetch = st.button(
        "🚀 Start Bulk Fetch and Update",
        type="primary",
        use_container_width=True,
        disabled=fetch_disabled,
        key="bulk_fetch_data_button"
    )

    if submitted_bulk_fetch and not fetch_disabled:
        fetch_args = argparse.Namespace()
        fetch_args.suggest_sources = (current_bulk_selection == "Use LLM Suggestion (Requires Topic)")
        fetch_args.topic = topic_value.strip() if topic_value else ""
        fetch_args.arxiv_query = custom_arxiv_query_value.strip() if custom_arxiv_query_value else ""
        fetch_args.num_arxiv = num_arxiv_results_value
        valid_input = True
        if fetch_args.suggest_sources and not fetch_args.topic:
            st.error("A topic is required when using the 'LLM Suggestion' method.", icon="❗")
            valid_input = False

        if valid_input:
            logger.info(f"Starting BULK data fetch with args: {fetch_args}")
            st.info("🚀 Starting bulk data fetch and processing pipeline... See terminal/logs for detailed progress.", icon="⏳")
            with st.spinner("Fetching, chunking, embedding, indexing... This may take a while depending on the amount of data."):
                try:
                    # setup_data_and_fetch implicitly uses crawl_and_fetch_web_articles
                    # which now defaults to processing linked PDFs.
                    status_message = setup_data_and_fetch(fetch_args)
                    if isinstance(status_message, str):
                        logger.info(f"Bulk fetch process completed with status: {status_message}")
                        if status_message.lower().startswith("success"):
                            st.success(f"Bulk Data Update Successful! {status_message}", icon="🎉")
                            st.info("Clearing component cache to load new data...")
                            st.session_state.force_component_reload = True
                            st.success("Cache cleared implicitly by forced reload. New data available.", icon="🔄")
                            st.rerun() # Rerun after successful bulk fetch
                        elif status_message.lower().startswith("error"):
                            st.error(f"Bulk Data Update Failed: {status_message}", icon="❌")
                        else:
                            st.warning(f"Bulk Data Update Status: {status_message}", icon="⚠️")
                    else:
                         st.warning(f"Received unexpected status type from bulk fetch process: {type(status_message)} - {status_message}", icon="⚠️")
                         logger.warning(f"Unexpected status type from bulk fetch process: {type(status_message)} - {status_message}")
                except Exception as e:
                    st.error(f"Critical error during bulk data fetch/processing: {e}", icon="🔥")
                    logger.error(f"Bulk Fetch/Processing Error in Streamlit App: {e}", exc_info=True)
                    st.code(traceback.format_exc())

# --- End Fetch Data Tab ---


# --- Search arXiv Tab ---
# (Remains unchanged)
with tab_arxiv:
    st.header("🔍 Direct arXiv Search")
    st.markdown("Perform a live search directly on the arXiv.org repository. This does *not* add data to the RAG knowledge base.")
    search_disabled = not project_modules_loaded
    if search_disabled:
        st.warning("Direct arXiv search disabled: Core project modules failed load.", icon="🚫")
    col1_arxiv, col2_arxiv = st.columns([4, 1])
    with col1_arxiv:
        arxiv_query = st.text_input("Search query for arXiv:", key="arxiv_query_input", label_visibility="collapsed", placeholder="Enter keywords, author (e.g., author:lecun), or title to search on arXiv...", disabled=search_disabled)
    with col2_arxiv:
        num_results = st.number_input("Max Results:", min_value=1, max_value=100, value=10, key="arxiv_num_input", disabled=search_disabled)
    if st.button("Search arXiv", key="arxiv_search_exec_button", type="secondary", use_container_width=True, disabled=search_disabled or not arxiv_query):
        if not arxiv_query: st.warning("Please enter an arXiv search query.")
        else:
            logger.info(f"Performing direct arXiv search for: '{arxiv_query}', max_results={num_results}")
            st.info(f"Searching arXiv.org for '{arxiv_query}'...", icon="📡")
            with st.spinner("Contacting arXiv API..."):
                try:
                    results = run_arxiv_search(arxiv_query, num_results)
                    if not results:
                        st.info("No results found on arXiv for this query.", icon="🤷")
                    else:
                        st.subheader(f"Found {len(results)} results on arXiv:")
                        st.markdown("---")
                        for i, paper in enumerate(results):
                            with st.container(border=True):
                                title = paper.get('title', 'N/A')
                                authors = ", ".join(paper.get('authors', [])) or 'N/A'
                                published = paper.get('published', 'N/A')
                                pdf_url = paper.get('pdf_url')
                                summary = paper.get('summary', 'N/A')
                                st.markdown(f"**{i + 1}. {title}**")
                                st.caption(f"👤 Authors: {authors} | 🗓️ Published: {published}")
                                if isinstance(pdf_url, str) and pdf_url.startswith('http'):
                                    st.link_button("View PDF 📄", pdf_url, help=f"Open PDF for '{title}'")
                                else:
                                    logger.warning(f"Invalid or missing PDF URL for paper '{title}': {pdf_url}")
                                    st.caption(" (PDF link unavailable)")
                                with st.expander("Show Abstract"):
                                    st.markdown(summary)
                        st.markdown("---")
                        st.success(f"Displayed {len(results)} results.", icon="✅")
                except Exception as e:
                    st.error(f"Error during arXiv search: {e}", icon="❌")
                    logger.error(f"arXiv Search Error: {e}", exc_info=True)
# --- End Search arXiv Tab ---


# --- Feedback Tab ---
# (Remains unchanged)
with tab_feedback:
    st.header("📝 Submit Feedback or Report Issues")
    st.markdown("""
    Your feedback is valuable for improving this tool! Please use the form below to share your thoughts,
    suggestions, or report any bugs you encounter.
    """)
    TALLY_ORIGINAL_EMBED_URL = "https://tally.so/embed/n0kkp6?alignLeft=1&hideTitle=1&transparentBackground=1&dynamicHeight=1"
    IFRAME_HEIGHT = 500
    logger.info(f"Embedding feedback form from Tally: {TALLY_ORIGINAL_EMBED_URL}")
    try:
        components.iframe(TALLY_ORIGINAL_EMBED_URL, height=IFRAME_HEIGHT, scrolling=True)
        st.caption("Feedback form securely hosted by Tally.so.")
    except Exception as e:
        st.error(f"Could not load the feedback form.", icon="😞")
        st.markdown("You can also submit feedback directly [here](https://tally.so/r/n0kkp6).")
        logger.error(f"Error embedding Tally iframe: {e}", exc_info=True)
# --- End Feedback Tab ---


# --- Footer ---
# (Remains unchanged)
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: grey; font-size: 0.9em;'>
        © 2025 Felix Nathaniel, Reynaldi Anatyo, Dennison Soedibjo | BINUS University Computer Science <br>
        Research Paper Assistant
    </div>
    """,
    unsafe_allow_html=True
)
# --- End Footer ---
