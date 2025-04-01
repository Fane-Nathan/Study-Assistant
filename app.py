# app.py
import streamlit as st
import nltk
import ssl
import os
import sys
import argparse
import traceback
import streamlit.components.v1 as components
import logging

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [App] %(message)s', force=True)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="Research Paper Assistant", # <--- UPDATED TITLE
    page_icon="📚",
    layout="wide"
)

# --- NLTK Data Download Logic ---
# (Code for checking/downloading NLTK data remains here - unchanged)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError: pass
else: ssl._create_default_https_context = _create_unverified_https_context
NLTK_PACKAGES = ['punkt', 'stopwords']
download_needed_flag = False
if 'nltk_data_checked' not in st.session_state: st.session_state.nltk_data_checked = False
if not st.session_state.nltk_data_checked:
    logger.info("Checking NLTK data...")
    with st.spinner("Checking/downloading NLTK data..."):
        all_packages_found = True
        # (Keep the loop checking and downloading NLTK packages here)
        for package in NLTK_PACKAGES:
             try:
                 nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
                 logger.info(f"NLTK package '{package}' found.")
             except LookupError:
                 logger.warning(f"NLTK package '{package}' not found. Triggering download.")
                 download_needed_flag = True
                 try:
                     nltk.download(package, quiet=True)
                     logger.info(f"NLTK '{package}' download attempt finished.")
                     nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
                     logger.info(f"NLTK '{package}' found after download attempt.")
                 except Exception as e:
                     st.error(f"Fatal Error: Failed to download NLTK package '{package}'. Error: {e}")
                     logger.error(f"NLTK '{package}' download failed: {e}", exc_info=True)
                     all_packages_found = False
                     st.stop()
             except Exception as E_find:
                  st.error(f"Fatal Error checking NLTK package '{package}'. Error: {E_find}")
                  logger.error(f"NLTK '{package}' check failed: {E_find}", exc_info=True)
                  all_packages_found = False
                  st.stop()
        # (Keep the rest of the NLTK check logic here)
        if download_needed_flag: logger.info("NLTK download process completed.")
        if all_packages_found:
            st.session_state.nltk_data_checked = True
            logger.info("NLTK session state marked as checked.")
        else:
             logger.error("Not all NLTK packages were found/downloaded successfully.")
             st.error("Failed to prepare all required NLTK packages.")
# --- End NLTK Data Download ---

# --- Path Setup ---
# (Path setup code remains here - unchanged)
logger.info("Setting up sys.path...")
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
logger.info(f"Project Root added to sys.path: {project_root}")
# --- End Path Setup ---

# --- Project Module Imports ---
# (Import logic remains here - unchanged)
logger.info("Attempting project module imports...")
project_modules_loaded = False
try:
    from hybrid_search_rag import config
    from scripts.cli import ( load_components, run_recommendation,
                              setup_data_and_fetch, run_arxiv_search )
    project_modules_loaded = True
    logger.info("Project module imports successful.")
except ImportError as e:
    st.error(f"Failed to import project modules (ImportError). Check setup. Error: {e}")
    st.code(f"sys.path: {sys.path}")
    logger.error(f"Project import failed (ImportError): {e}", exc_info=True)
except Exception as e:
    st.error(f"Unexpected error during project imports: {e}. App limited.")
    logger.error(f"Project import failed (Exception): {e}", exc_info=True)
# --- End Project Module Imports ---

# --- Title, Introduction, and Disclaimer ---
# (Title, intro, and disclaimer remain here - unchanged)
st.title("📚 Research Paper Assistant")
st.markdown("""
Explore academic research effortlessly. This tool uses advanced AI (RAG & Hybrid Search)
to find relevant information, answer your questions from research papers, and provide cited summaries.
""")
st.info("""
**Disclaimer:** This tool is designed to assist in finding and ranking relevant research papers based on your queries.
It is **not** intended as a replacement for general-purpose AI chatbots (like ChatGPT, Gemini, etc.) for conversational tasks or broad knowledge questions.
""", icon="ℹ️")
st.divider()
# --- End Title/Intro/Disclaimer ---

# --- Caching Components ---
# (Cached component loading function remains here - unchanged)
@st.cache_resource
def cached_load_components():
    logger.info("Attempting load via cached_load_components...")
    if not project_modules_loaded:
        logger.error("Cannot load components: project modules failed import.")
        return False
    try:
        load_components(force_reload=False)
        from scripts.cli import loaded_metadata
        if loaded_metadata is not None:
             logger.info("Components loaded successfully.")
             return True
        else:
             logger.warning("load_components ran but indicator 'loaded_metadata' is None.")
             st.warning("Components appear missing after loading attempt.")
             return False
    except NameError as ne:
         st.error(f"Error accessing loaded state from 'scripts.cli'. Error: {ne}")
         logger.error(f"NameError during component loading: {ne}", exc_info=True)
         return False
    except Exception as e:
        st.error(f"Error loading RAG components: {e}")
        logger.error(f"Exception during component loading: {e}", exc_info=True)
        return False
# --- End Caching Components ---

# --- Initial Setup ---
# (Initial component load call remains here - unchanged)
logger.info("Calling cached_load_components...")
components_loaded = cached_load_components() if project_modules_loaded else False
logger.info(f"Components loaded status: {components_loaded}")
if not components_loaded and project_modules_loaded:
    st.error("Core RAG components failed to load. Recommendation functionality disabled.")

# --- UI Tabs ---
# (Tab definition remains here - unchanged)
logger.info("Defining UI tabs...")
tab_rec, tab_how, tab_about, tab_fetch, tab_arxiv, tab_feedback = st.tabs([
    "🧠 Recommend", "⚙️ How It Works", "ℹ️ About",
    "⏬ Fetch Data", "🔍 Search arXiv", "📝 Feedback"
])
logger.info("UI tabs defined.")

# --- Recommendation Tab ---
# (Recommend tab logic remains here - unchanged, including the expander fix)
with tab_rec:
    st.header("Ask the RAG Assistant")
    st.markdown("Enter your topic or question below and get a cited answer based on the indexed documents.")
    col1_rec, col2_rec = st.columns([3, 1])
    with col1_rec:
        default_query = config.DEFAULT_QUERY if project_modules_loaded else "Example: What is RAG?"
        query = st.text_area("Your Question:", value=st.session_state.get("rec_query", default_query), height=150, key="rec_query_input_area", label_visibility="collapsed")
    with col2_rec:
        general_mode = st.checkbox("Hybrid Mode", value=st.session_state.get("rec_general", False), key="rec_general_toggle", help="Allow LLM general knowledge.")
        concise_mode = st.checkbox("Concise Prompt", value=st.session_state.get("rec_concise", True), key="rec_concise_toggle", help="Use structured, concise prompt.")
        submit_rec = st.button("Get Recommendation", type="primary", key="rec_button", disabled=not components_loaded, use_container_width=True)

    if submit_rec:
        if not query: st.warning("Please enter a question.")
        elif not components_loaded: st.error("Cannot recommend: Core components failed load.")
        else:
            st.session_state.rec_query = query
            st.session_state.rec_general = general_mode
            st.session_state.rec_concise = concise_mode
            mode_name = "Hybrid" if general_mode else "Strict"
            style_name = "Concise" if concise_mode else "Detailed"
            st.info(f"Running recommendation ({mode_name} RAG, {style_name} Prompt)...")
            with st.spinner("Retrieving context and generating response..."):
                try:
                    top_n = config.TOP_N_RESULTS if project_modules_loaded else 5
                    llm_answer, context_sources = run_recommendation(query, top_n, general_mode, concise_mode)
                    st.session_state.llm_answer = llm_answer
                    st.session_state.context_sources = context_sources
                except Exception as e:
                    st.error(f"Error during recommendation: {e}")
                    logger.error(f"Recommendation Error: {e}", exc_info=True)
                    st.session_state.llm_answer = None
                    st.session_state.context_sources = []

    # Display results
    if "llm_answer" in st.session_state and st.session_state.llm_answer:
        st.markdown("---")
        st.subheader("Assistant Response")
        st.markdown(st.session_state.llm_answer, unsafe_allow_html=True)

    if "context_sources" in st.session_state:
         st.markdown("---")
         with st.expander("Show Context Sources Provided to LLM"):
             sources = st.session_state.context_sources
             if sources:
                 # Join with two spaces and newline for correct markdown line breaks
                 formatted_sources = "  \n".join(sources)
                 st.markdown(formatted_sources)
             else:
                 st.info("No relevant local document chunks were provided as context for this answer.")
# --- End Recommendation Tab ---


# --- How It Works Tab ---
# *** MODIFIED FOR BETTER MOBILE VIEW ***
with tab_how:
    st.header("How the RAG System Works")
    st.markdown("This system uses a Retrieval-Augmented Generation (RAG) pipeline:")
    # Removed the 3-column layout for better stacking on mobile

    st.subheader("1. Data Ingestion & Processing")
    st.markdown("""
    * Fetches data from arXiv, web URLs, and PDFs (using PyMuPDF).
    * Cleans text and applies sentence-based chunking.
    * Breaks down long documents for better analysis.
    """)

    st.subheader("2. Hybrid Indexing")
    emb_model = config.EMBEDDING_MODEL_NAME if project_modules_loaded else "an embedding model"
    st.markdown(f"""
    * Generates dense vector embeddings (using `{emb_model}`) for semantic understanding.
    * Builds a sparse keyword index (BM25) for term matching.
    * Indexes are persisted locally.
    """)

    st.subheader("3. Hybrid Retrieval (RRF)")
    st.markdown("""
    * Queries both embedding and BM25 indexes simultaneously.
    * Combines results using Reciprocal Rank Fusion (RRF).
    * Produces a single, relevance-ranked list of document chunks.
    """)

    st.subheader("4. LLM Generation & Citation")
    st.markdown("""
    * Retrieved chunks are passed to a large language model (e.g., Gemini).
    * LLM generates a coherent answer based on context (Strict RAG) or context + general knowledge (Hybrid Mode).
    * Citations `[N]` are automatically formatted based on instructions.
    """)

    st.subheader("5. RAG Pipeline")
    st.markdown("""
    * The entire process forms a RAG pipeline.
    * Enhances LLM responses with specific, retrieved information.
    * Improves factual grounding and reduces hallucinations.
    """)

    st.subheader("6. Core Implementation")
    st.markdown("""
    * Built using Python with modular components.
    * Features include dynamic source finding, chunking, hybrid search, and configurable LLM integration.
    * This Streamlit app provides the UI.
    """)
    st.divider()
# --- End How It Works Tab ---


# --- About Tab ---
# *** MODIFIED FOR BETTER MOBILE VIEW ***
with tab_about:
    st.header("About This Project")
    # Removed the 2-column layout

    st.subheader("Project Goal")
    st.markdown("""
    To develop a tool assisting users in navigating, understanding, and discovering academic information,
    primarily focusing on Machine Learning research papers, with future goals for personalization and
    resource recommendation.
    """)

    st.subheader("Current Status")
    st.markdown("""
    The project functions as a sophisticated RAG tool capable of ingesting various document types
    (including PDFs via crawling), building a chunked hybrid index, and accurately answering questions
    with citations based on the indexed content. Key features like hybrid search, chunking,
    LLM integration (Gemini), and configurable RAG modes are implemented.
    """)

    st.subheader("Future Work")
    st.markdown("""
    * **Expand Corpus:** Build a larger, diverse, curated knowledge base.
    * **Tune Retrieval:** Systematically evaluate and optimize retrieval performance.
    * **Implement Recommender Features:** Add proactive topic suggestions.
    * **Add Personalization:** Incorporate user profiles, history, etc.
    * **Refine UI/UX:** Continuously improve the web interface.
    * **(Maybe) API Endpoint:** Create a dedicated API for broader integration.
    """)

    st.subheader("Developer")
    st.markdown(f"""
    This project is developed by **Felix Nathaniel**, **Reynaldi Anatyo**, & **Dennison Soebdibjo** a Computer Science student at BINUS University,
    as part of ongoing learning and exploration in AI/ML.
    """)
    st.caption("*(Powered by Streamlit)*") # Moved caption slightly
    st.divider()
# --- End About Tab ---


# --- Fetch Data Tab ---
# (Fetch Data tab logic remains here - unchanged)
with tab_fetch:
    st.header("Update Knowledge Base")
    st.warning("Fetching and processing can take significant time and requires internet access.", icon="⏳")
    fetch_disabled = not project_modules_loaded
    if fetch_disabled: st.warning("Data fetching disabled: core modules failed to load.")

    with st.form("fetch_form"):
        st.subheader("Source Selection")
        fetch_method = st.radio(
            "Select Fetch Method:",
            ("Use Defaults (from config.py)", "Use Custom arXiv Query", "Use LLM Suggestion (Requires Topic)"),
            key="fetch_method", horizontal=True, disabled=fetch_disabled
        )
        col1_fetch, col2_fetch = st.columns(2) # Keeping columns here might be okay for input layout
        with col1_fetch:
            custom_arxiv_query = st.text_input("Custom arXiv Query:", key="fetch_arxiv_query", help="Used if 'Custom arXiv Query' is selected.", disabled=fetch_disabled)
            topic = st.text_input("Topic for LLM Suggestion:", key="fetch_topic", help="Used if 'LLM Suggestion' is selected.", disabled=fetch_disabled)
        with col2_fetch:
            default_arxiv_results = config.MAX_ARXIV_RESULTS if project_modules_loaded else 10
            num_arxiv_results = st.number_input(f"Max arXiv Results:", min_value=5, max_value=500, value=default_arxiv_results, key="fetch_num_arxiv", help="Applies to default, custom, or suggested query.", disabled=fetch_disabled)
        st.divider()
        submitted_fetch = st.form_submit_button("Fetch and Process Data", type="primary", use_container_width=True, disabled=fetch_disabled)

        if submitted_fetch and not fetch_disabled:
            fetch_args = argparse.Namespace()
            fetch_args.suggest_sources = (fetch_method == "Use LLM Suggestion (Requires Topic)")
            fetch_args.topic = topic if fetch_args.suggest_sources else None
            fetch_args.arxiv_query = custom_arxiv_query if fetch_method == "Use Custom arXiv Query" else None
            fetch_args.num_arxiv = num_arxiv_results
            valid_input = True
            if fetch_args.suggest_sources and not fetch_args.topic:
                st.error("Topic required for LLM Suggestion.")
                valid_input = False
            if fetch_method == "Use Custom arXiv Query" and not fetch_args.arxiv_query:
                st.error("Custom arXiv Query required.")
                valid_input = False

            if valid_input:
                st.info("Starting data fetch... See terminal for logs.")
                with st.spinner("Fetching, chunking, embedding, indexing... Please wait."):
                    try:
                        status_message = setup_data_and_fetch(fetch_args)
                        if status_message.lower().startswith("success"):
                            st.success(f"Fetch Successful: {status_message}")
                            st.info("Clearing component cache to reload data...")
                            cached_load_components.clear()
                            st.rerun()
                        elif status_message.lower().startswith("error"): st.error(f"Fetch Failed: {status_message}")
                        else: st.warning(f"Fetch Status: {status_message}")
                    except Exception as e:
                        st.error(f"Critical error during fetch/processing: {e}")
                        logger.error(f"Fetch Error: {e}", exc_info=True)
# --- End Fetch Data Tab ---


# --- Search arXiv Tab ---
# (Search arXiv tab logic remains here - unchanged)
with tab_arxiv:
    st.header("Direct arXiv Search")
    st.markdown("Perform a live search directly on arXiv.org.")
    search_disabled = not project_modules_loaded
    if search_disabled: st.warning("Direct arXiv search disabled: core modules failed load.")

    col1_arxiv, col2_arxiv = st.columns([3, 1]) # Keeping columns for input layout
    with col1_arxiv:
        arxiv_query = st.text_input("Search query for arXiv:", key="arxiv_query_input", label_visibility="collapsed", placeholder="Enter arXiv search query...", disabled=search_disabled)
    with col2_arxiv:
        num_results = st.number_input("Max Results:", min_value=1, max_value=100, value=10, key="arxiv_num_input", label_visibility="collapsed", disabled=search_disabled)

    if st.button("Search arXiv", key="arxiv_search_exec_button", type="primary", use_container_width=True, disabled=search_disabled):
        if not arxiv_query: st.warning("Please enter an arXiv search query.")
        else:
            st.info(f"Searching arXiv for '{arxiv_query}'...")
            with st.spinner("Searching arXiv..."):
                try:
                    results = run_arxiv_search(arxiv_query, num_results)
                    if not results: st.info("No results found on arXiv.")
                    else:
                        st.subheader(f"Found {len(results)} results:")
                        for i, paper in enumerate(results):
                            st.markdown(f"**{i + 1}. {paper.get('title', 'N/A')}**")
                            st.caption(f"Authors: {', '.join(paper.get('authors', [])) or 'N/A'} | Published: {paper.get('published', 'N/A')}")
                            pdf_url = paper.get('pdf_url')
                            if isinstance(pdf_url, str) and pdf_url.startswith('http'):
                                st.link_button("View PDF", pdf_url)
                            else: logger.warning(f"Invalid PDF URL for paper '{paper.get('title', 'N/A')}': {pdf_url}")
                            with st.expander("Show Abstract"):
                                st.write(paper.get('summary', 'N/A'))
                            st.divider()
                except Exception as e:
                    st.error(f"Error during arXiv search: {e}")
                    logger.error(f"arXiv Search Error: {e}", exc_info=True)
# --- End Search arXiv Tab ---


# --- Feedback Tab ---
# (Feedback tab logic remains here - unchanged)
with tab_feedback:
    st.header("Submit Feedback")
    TALLY_ORIGINAL_EMBED_URL = "https://tally.so/embed/n0kkp6?alignLeft=1&hideTitle=1&transparentBackground=1&dynamicHeight=1"
    IFRAME_HEIGHT = 500
    logger.info(f"Embedding feedback form from Tally: {TALLY_ORIGINAL_EMBED_URL}")
    try:
        components.iframe(TALLY_ORIGINAL_EMBED_URL, height=IFRAME_HEIGHT, scrolling=True)
        st.caption("Feedback form provided by Tally.so.")
    except Exception as e:
        st.error(f"Failed to load feedback form iframe.")
        logger.error(f"Error embedding Tally iframe: {e}", exc_info=True)
# --- End Feedback Tab ---


# --- Footer ---
# (Footer remains here - unchanged)
st.divider()
st.markdown("<div style='text-align: center; color: grey;'>© 2025 Felix Nathaniel, Reynaldi Anatyo, Dennison Soebdibjo | Research Paper Assistant</div>", unsafe_allow_html=True)