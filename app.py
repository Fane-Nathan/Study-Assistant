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
import math

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [App] %(message)s', force=True)
logger = logging.getLogger(__name__)

# --- Page Configuration (Enhanced) ---
st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚", # Good choice!
    layout="wide",  # Keep 'wide' layout for more space
    initial_sidebar_state="auto", # Allow sidebar usage if we add one later
    menu_items={ # Add useful menu items
        'Get Help': 'https://github.com/Fane-Nathan/Study-Assistant', # Link to your repo or help page
        'Report a bug': "https://tally.so/r/n0kkp6", # Link directly to feedback/bug report
        'About': "# About This Project\nThis app helps explore academic research using RAG."
    }
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
    # Use st.progress for visual feedback during NLTK download check
    progress_bar = st.progress(0, text="Checking NLTK data requirements...")
    with st.spinner("Verifying core language libraries..."):
        all_packages_found = True
        total_packages = len(NLTK_PACKAGES)
        for i, package in enumerate(NLTK_PACKAGES):
             progress_text = f"Checking NLTK package: '{package}' ({i+1}/{total_packages})..."
             progress_bar.progress((i + 1) / total_packages, text=progress_text)
             try:
                 nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
                 logger.info(f"NLTK package '{package}' found.")
             except LookupError:
                 logger.warning(f"NLTK package '{package}' not found. Triggering download.")
                 download_needed_flag = True
                 progress_bar.progress((i + 1) / total_packages, text=f"Downloading NLTK package: '{package}'...")
                 try:
                     nltk.download(package, quiet=True)
                     logger.info(f"NLTK '{package}' download attempt finished.")
                     nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
                     logger.info(f"NLTK '{package}' found after download attempt.")
                 except Exception as e:
                     st.error(f"Fatal Error: Failed to download NLTK package '{package}'. Error: {e}")
                     logger.error(f"NLTK '{package}' download failed: {e}", exc_info=True)
                     all_packages_found = False
                     progress_bar.empty() # Clear progress bar on failure
                     st.stop()
             except Exception as E_find:
                  st.error(f"Fatal Error checking NLTK package '{package}'. Error: {E_find}")
                  logger.error(f"NLTK '{package}' check failed: {E_find}", exc_info=True)
                  all_packages_found = False
                  progress_bar.empty() # Clear progress bar on failure
                  st.stop()
        # (Keep the rest of the NLTK check logic here)
        progress_bar.empty() # Clear progress bar on success
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

# --- Title, Introduction, and Disclaimer (Enhanced) ---
st.title("📚 Research Paper Assistant")
st.subheader("Unlock insights from academic literature with AI") # Added a subheader for context
st.markdown("""
Explore academic research effortlessly. This tool leverages Retrieval-Augmented Generation (RAG)
and Hybrid Search to find relevant papers, answer your questions using document context, and provide cited summaries.
""")

# Use st.expander for the disclaimer to make it less visually dominant initially
with st.expander("ℹ️ Important Disclaimer", expanded=False):
    st.warning("""
    **Please Note:** This tool is specialized for finding and analyzing information within the indexed research papers.
    It excels at providing context-aware, cited answers based on its knowledge base.

    It is **not** a general-purpose chatbot (like ChatGPT, Gemini, etc.) and may not perform well on:
    * General knowledge questions outside the scope of the indexed documents.
    * Creative writing or conversational tasks.
    * Real-time information (e.g., news, weather).
    """, icon="⚠️")
st.divider()
# --- End Title/Intro/Disclaimer ---

# --- Caching Components ---
# (Cached component loading function remains here - unchanged)
@st.cache_resource # Keep the decorator for the expensive loading
def cached_load_components():
    # This function now ONLY performs the loading and returns status.
    # NO Streamlit UI elements (spinner, toast, error, warning) should be called inside here.
    logger.info("Executing RAG component loading logic...")
    if not project_modules_loaded:
        logger.error("Cannot load components: project modules failed import prior to this call.")
        # Return failure status and an informative message
        return False, "Core project modules failed to import."
    try:
        # *** Perform your actual loading here ***
        load_components(force_reload=False)

        # *** Check if loading was successful (using your indicator) ***
        from scripts.cli import loaded_metadata # Assuming this indicates successful load
        if loaded_metadata is not None: # Replace with your actual check
             logger.info("Component loading logic successful.")
             # Return success status and no error message
             return True, None
        else:
             logger.warning("load_components ran but indicator 'loaded_metadata' is None.")
             # Return failure status and an informative message
             return False, "Components appear missing after loading attempt (indicator is None)."
    except NameError as ne:
         # Log the error internally
         logger.error(f"NameError during component loading: {ne}", exc_info=True)
         # Return failure status and the error message
         return False, f"Error accessing loaded state from 'scripts.cli'. NameError: {ne}"
    except Exception as e:
        # Log the error internally
        logger.error(f"Exception during component loading: {e}", exc_info=True)
        # Return failure status and the error message
        return False, f"An exception occurred during component loading: {e}"
# --- End Caching Components ---

# --- Initial Setup ---
# (Initial component load call remains here - unchanged)
logger.info("Calling cached_load_components...")
components_loaded = cached_load_components() if project_modules_loaded else False
logger.info(f"Components loaded status: {components_loaded}")
if not components_loaded and project_modules_loaded:
    st.error("Core RAG components failed to load. Recommendation and Fetch Data functionality disabled.")

# --- UI Tabs (with Icons) ---
logger.info("Defining UI tabs...")
# Add relevant emojis to tab names for better visual distinction
tab_rec, tab_how, tab_about, tab_fetch, tab_arxiv, tab_feedback = st.tabs([
    "🧠 **Recommend**", "⚙️ How It Works", "ℹ️ About",
    "⏬ Update Knowledge Base", "🔍 Search arXiv", "📝 Feedback"
])
logger.info("UI tabs defined.")

# --- Recommendation Tab (Enhanced) ---
with tab_rec:
    st.header("💬 Ask the RAG Assistant")
    st.markdown("Enter your research topic or question below. The assistant will retrieve relevant information from the indexed documents and generate a cited answer.")

    col1_rec, col2_rec = st.columns([3, 1]) # Keep layout for input/controls

    with col1_rec:
        default_query = config.DEFAULT_QUERY if project_modules_loaded else "Example: Explain the core concepts of Retrieval-Augmented Generation (RAG)."
        query = st.text_area(
            "Your Question:",
            value=st.session_state.get("rec_query", default_query),
            height=150,
            key="rec_query_input_area",
            placeholder="Type your question about the research papers here...",
            help="Ask anything related to the content of the indexed documents.",
            label_visibility="collapsed" # Cleaner look if header is clear
        )

    with col2_rec:
        st.markdown("**Options**") # Add a small title for options
        general_mode = st.toggle( # Use toggle for a more modern feel than checkbox
            "Hybrid Mode",
            value=st.session_state.get("rec_general", False),
            key="rec_general_toggle",
            help="Allows the AI to use its general knowledge *in addition* to the retrieved documents. 'Strict Mode' (off) uses *only* document context."
        )
        concise_mode = st.toggle( # Use toggle here too
            "Concise Prompt",
            value=st.session_state.get("rec_concise", True),
            key="rec_concise_toggle",
            help="Uses a more structured, concise prompt for the AI, potentially faster but less conversational."
        )
        st.markdown("---") # Visual separator
        submit_rec = st.button(
            "✨ Get Recommendation", # Add sparkle
            type="primary",
            key="rec_button",
            disabled=not components_loaded or not query, # Disable if no query OR components not loaded
            use_container_width=True
        )

    if submit_rec:
        # Removed redundant 'if not query' check because button is disabled
        if not components_loaded: st.error("Cannot recommend: Core components failed load.")
        else:
            st.session_state.rec_query = query
            st.session_state.rec_general = general_mode
            st.session_state.rec_concise = concise_mode
            mode_name = "Hybrid" if general_mode else "Strict"
            style_name = "Concise" if concise_mode else "Detailed"
            st.info(f"Running recommendation with '{mode_name}' RAG and '{style_name}' Prompt...", icon="⏳")
            # Use st.spinner for better visual feedback during processing
            with st.spinner("🧠 Thinking... Retrieving context and generating response..."):
                try:
                    top_n = config.TOP_N_RESULTS if project_modules_loaded else 5
                    llm_answer, context_sources = run_recommendation(query, top_n, general_mode, concise_mode)
                    st.session_state.llm_answer = llm_answer
                    st.session_state.context_sources = context_sources
                    st.toast("Response generated!", icon="✅")
                except Exception as e:
                    st.error(f"Error during recommendation: {e}", icon="❌")
                    logger.error(f"Recommendation Error: {e}", exc_info=True)
                    st.session_state.llm_answer = None
                    st.session_state.context_sources = []

    if "llm_answer" in st.session_state and st.session_state.llm_answer:
        st.markdown("---")
        st.subheader("Assistant's Response")
        st.markdown(st.session_state.llm_answer, unsafe_allow_html=True)

    if "context_sources" in st.session_state:

         if st.session_state.context_sources or ("llm_answer" in st.session_state and st.session_state.llm_answer):
            st.markdown("---")

            with st.expander("📚 Show Context Sources Provided to LLM", expanded=False):
                 sources = st.session_state.context_sources
                 if sources:
                     st.caption("These are the document snippets retrieved and used to generate the answer above:")
                     num_sources = len(sources)

                     if num_sources > 0:
                         num_columns = 2
                         cols = st.columns(num_columns)
                         midpoint = math.ceil(num_sources / num_columns)

                         with cols[0]:
                             for i in range(midpoint):
                                 if i < num_sources:
                                     st.markdown(f"{sources[i]}")

                         with cols[1]:
                             for i in range(midpoint, num_sources):
                                 st.markdown(f"{sources[i]}")

                 else:
                     st.info("No specific document chunks were retrieved as relevant context for this query.")


# --- How It Works Tab (Enhanced with Icons/Formatting) ---
with tab_how:
    st.header("⚙️ How the RAG System Works")
    st.markdown("This system uses a **Retrieval-Augmented Generation (RAG)** pipeline to provide informed answers:")

    col1_how, col2_how = st.columns(2)

    with col1_how:
        st.subheader("1️⃣ Data Ingestion & Processing")
        st.markdown("""
        * **Sources:** Fetches data from arXiv, web URLs, local PDFs.
        * **Cleaning:** Prepares text for analysis.
        * **Chunking:** Breaks down documents into smaller, manageable pieces (e.g., by sentence or paragraph) for effective retrieval.
        """)

        st.subheader("2️⃣ Hybrid Indexing")
        emb_model = f"`{config.EMBEDDING_MODEL_NAME}`" if project_modules_loaded else "a sentence transformer"
        st.markdown(f"""
        * **Vector Embeddings:** Creates numerical representations (vectors) capturing semantic meaning using {emb_model}. Allows finding *conceptually similar* text.
        * **Keyword Index (BM25):** Creates a traditional keyword index to find exact term matches.
        * **Combined Power:** Stores both index types locally for fast lookup.
        """)

    with col2_how:
        st.subheader("3️⃣ Hybrid Retrieval (RRF)")
        st.markdown("""
        * **Dual Search:** Your query searches *both* the vector index (for meaning) and the keyword index (for terms) simultaneously.
        * **Smart Ranking (RRF):** Results from both searches are combined using Reciprocal Rank Fusion (RRF) to produce a single, relevance-ranked list of document chunks. This balances semantic relevance and keyword importance.
        """)

        st.subheader("4️⃣ LLM Generation & Citation")
        llm_name = f"{config.LLM_PROVIDER}" if project_modules_loaded else "a Large Language Model (LLM)"
        st.markdown(f"""
        * **Context Injection:** The top-ranked retrieved chunks are passed as context to {llm_name} (e.g., Gemini).
        * **Informed Answering:** The LLM generates a coherent answer *based on the provided context* (Strict RAG) or a mix of context and its internal knowledge (Hybrid Mode).
        * **Citation:** The system attempts to add citations `[N]` linking parts of the answer back to the specific source chunks.
        """)

    st.divider()

    st.subheader("🚀 The RAG Pipeline Advantage")
    st.markdown("""
    This entire process forms the RAG pipeline. It enhances standard LLM responses by:
    * Grounding answers in specific, retrieved information from your documents.
    * Reducing the likelihood of the LLM making up information ("hallucinations").
    * Providing transparency through source citations.
    """)

    st.subheader("💻 Core Implementation")
    st.markdown("""
    * Built using Python with libraries like `LangChain`, `SentenceTransformers`, `Rank_BM25`, `FAISS` (or similar vector stores), and `Streamlit`.
    * Features modular components for data handling, indexing, retrieval, and generation.
    """)
    st.divider()
# --- End How It Works Tab ---


# --- About Tab (Enhanced with Icons/Formatting) ---
with tab_about:
    st.header("ℹ️ About This Project")

    # Using columns here can also work, similar to 'How It Works'
    col1_about, col2_about = st.columns(2)

    with col1_about:
        st.subheader("🎯 Project Goal")
        st.markdown("""
        To develop an intelligent assistant that helps users efficiently navigate, understand, and discover information within a specific corpus of academic documents (initially focusing on AI/ML research).
        """)

        st.subheader("🛠️ Current Status & Features")
        st.markdown("""
        * **RAG Core:** Functional RAG pipeline implemented.
        * **Data Sources:** Ingests arXiv papers, web links, and PDFs.
        * **Hybrid Search:** Combines semantic (vector) and keyword (BM25) search with RRF.
        * **LLM Integration:** Uses Gemini for generation, with configurable modes (Strict/Hybrid).
        * **Citation:** Provides source citations in responses.
        * **UI:** Interactive Streamlit interface for querying, data fetching, and arXiv search.
        """)

    with col2_about:
        st.subheader("🚀 Future Work & Ideas")
        st.markdown("""
        * **Corpus Expansion:** Curate and index a larger, more diverse set of relevant papers. 
        * **Evaluation & Tuning:** Systematically evaluate retrieval and generation quality.
        * **Recommender System:** Proactively suggest relevant papers or topics based on user queries or profiles
        * **Personalization:** Allow user profiles, history tracking, and preference settings.
        * **Deployment:** Explore options for more robust deployment.
        """)

        st.subheader("👨‍💻 Developers")
        st.markdown(f"""
        Developed by **Felix Nathaniel**, **Reynaldi Anatyo**, & **Dennison Soedibjo**.

        *Computer Science Students at BINUS University, exploring the fascinating world of AI and Information Retrieval.*
        """)
        st.link_button("View Project on GitHub", "https://github.com/Fane-Nathan/Study-Assistant") # Add link button
# --- End About Tab ---


# --- Fetch Data Tab (Enhanced UI) ---
with tab_fetch:
    st.header("⏬ Update Knowledge Base")
    st.markdown("Add new documents to the assistant's knowledge base from various sources.")
    st.warning("Fetching and processing new data can take significant time and requires a stable internet connection.", icon="⏳")

    fetch_disabled = not project_modules_loaded
    if fetch_disabled:
        st.error("Data fetching disabled: Core project modules failed to load.", icon="🚫")

    # --- Widgets OUTSIDE the form ---
    st.subheader("Select Data Source Method")
    fetch_options = ("Use Custom arXiv Query", "Use LLM Suggestion (Requires Topic)")

    # Radio button - determines which input below is active
    fetch_method = st.radio(
        "Choose how to find documents:",
        options=fetch_options,
        index=0,  # Default to 'Use Custom arXiv Query'
        key="fetch_method_radio", # Use a unique key
        horizontal=True,
        disabled=fetch_disabled,
    )

    st.markdown("---") # Separator

    # Get the current selection reliably (needed for disabling logic below)
    # Access session state directly using the key. Default to the first option if not set.
    current_selection = st.session_state.get("fetch_method_radio", fetch_options[0])

    col1_fetch, col2_fetch = st.columns(2)

    with col1_fetch:
        # Custom Query Text Input - Disable if LLM Suggestion is selected
        custom_arxiv_query_value = st.text_input(
            "Custom arXiv Query:",
            key="fetch_arxiv_query_val", # Unique key
            placeholder="e.g., 'attention mechanism transformer'",
            help="Enter keywords or query for arXiv.",
            # Disable based on the *current selection* of the radio button
            disabled=fetch_disabled or (current_selection == "Use LLM Suggestion (Requires Topic)")
        )

        # Topic Text Input - Disable if Custom Query is selected
        topic_value = st.text_input(
            "Topic for LLM Suggestion:",
            key="fetch_topic_val", # Unique key
            placeholder="e.g., 'Latest advancements in RAG'",
            help="Enter a topic if 'LLM Suggestion' is selected.",
            # Disable based on the *current selection* of the radio button
            disabled=fetch_disabled or (current_selection == "Use Custom arXiv Query")
        )

    with col2_fetch:
        # Number input - Can stay outside form
        num_arxiv_results_value = st.number_input(
            f"Max arXiv Results to Fetch:",
            min_value=5,
            max_value=500,
            # Use a default value directly here
            value=st.session_state.get("fetch_num_results_val", config.MAX_ARXIV_RESULTS if project_modules_loaded else 10),
            key="fetch_num_results_val", # Unique key
            help="Maximum number of papers to retrieve from arXiv for processing.",
            disabled=fetch_disabled
        )

    st.markdown("---") # Separator before submit button

    # --- Button OUTSIDE the form ---
    submitted_fetch = st.button( # Regular button, not form submit
        "Fetch and Add Data",
        type="primary",
        use_container_width=True,
        disabled=fetch_disabled,
        key="fetch_data_button" # Unique key
    )

    # --- Validation and Fetch logic ---
    # This runs ONLY when the button is clicked
    if submitted_fetch and not fetch_disabled:
        # Read values directly from the variables returned by the widgets
        fetch_args = argparse.Namespace()
        fetch_args.suggest_sources = (current_selection == "Use LLM Suggestion (Requires Topic)")
        fetch_args.topic = topic_value
        fetch_args.arxiv_query = custom_arxiv_query_value
        fetch_args.num_arxiv = num_arxiv_results_value

        valid_input = True
        if fetch_args.suggest_sources and not fetch_args.topic:
            st.error("A topic is required when using the 'LLM Suggestion' method.", icon="❗")
            valid_input = False
        elif not fetch_args.suggest_sources and not fetch_args.arxiv_query:
             st.error("An arXiv query is required when using the 'Custom arXiv Query' method.", icon="❗")
             valid_input = False

        if valid_input:
            # Optional: Clear the inactive field's value from session state if desired
            # if fetch_args.suggest_sources:
            #     st.session_state.fetch_arxiv_query_val = ""
            # else:
            #     st.session_state.fetch_topic_val = ""

                st.info("🚀 Starting data fetch and processing pipeline... See terminal/logs for detailed progress.", icon="⏳")
                # Use a spinner for the duration of the operation
                with st.spinner("Fetching, chunking, embedding, indexing... This may take a while depending on the amount of data."):
                    try:
                        # Pass necessary arguments, including potentially uploaded files
                        status_message = setup_data_and_fetch(fetch_args) # Make sure this function accepts the args
                        if isinstance(status_message, str): # Check if status is string
                            if status_message.lower().startswith("success"):
                                st.success(f"Data Update Successful! {status_message}", icon="🎉")
                                st.info("Clearing component cache to load new data...")
                                cached_load_components.clear() # Crucial step!
                                st.success("Cache cleared. Rerun the app or switch tabs to see updates.", icon="🔄")
                                # st.rerun() # Force rerun might be too abrupt, let user navigate
                            elif status_message.lower().startswith("error"):
                                st.error(f"Data Update Failed: {status_message}", icon="❌")
                            else:
                                st.warning(f"Data Update Status: {status_message}", icon="⚠️")
                        else:
                             st.warning(f"Received unexpected status from fetch process: {status_message}", icon="⚠️")

                    except Exception as e:
                        st.error(f"Critical error during data fetch/processing: {e}", icon="🔥")
                        logger.error(f"Fetch Error: {e}", exc_info=True)
                        st.code(traceback.format_exc()) # Show traceback in UI for debugging
# --- End Fetch Data Tab ---


# --- Search arXiv Tab (Enhanced Results Display) ---
with tab_arxiv:
    st.header("🔍 Direct arXiv Search")
    st.markdown("Perform a live search directly on the arXiv.org repository. This does *not* add data to the RAG knowledge base.")

    search_disabled = not project_modules_loaded
    if search_disabled:
        st.warning("Direct arXiv search disabled: Core project modules failed load.", icon="🚫")

    # Keep layout for search input and controls
    col1_arxiv, col2_arxiv = st.columns([4, 1]) # Give query slightly more space

    with col1_arxiv:
        arxiv_query = st.text_input(
            "Search query for arXiv:",
            key="arxiv_query_input",
            label_visibility="collapsed",
            placeholder="Enter keywords, author, or title to search on arXiv...",
            disabled=search_disabled
        )
    with col2_arxiv:
        num_results = st.number_input(
            "Max Results:",
            min_value=1,
            max_value=100,
            value=10,
            key="arxiv_num_input",
            # label_visibility="collapsed", # Keep label for clarity
            disabled=search_disabled
        )

    if st.button("Search arXiv", key="arxiv_search_exec_button", type="secondary", use_container_width=True, disabled=search_disabled or not arxiv_query): # Disable if no query
        if not arxiv_query: st.warning("Please enter an arXiv search query.") # Should not happen if button disabled
        else:
            st.info(f"Searching arXiv.org for '{arxiv_query}'...", icon="📡")
            with st.spinner("Contacting arXiv API..."):
                try:
                    results = run_arxiv_search(arxiv_query, num_results) # Assumes this returns a list of dicts

                    if not results:
                        st.info("No results found on arXiv for this query.", icon="🤷")
                    else:
                        st.subheader(f"Found {len(results)} results on arXiv:")
                        st.markdown("---")
                        # Iterate and display each result in a container for visual separation
                        for i, paper in enumerate(results):
                            with st.container(border=True): # Add border around each result
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
                                    logger.warning(f"Invalid PDF URL for paper '{title}': {pdf_url}")
                                    st.caption(" (PDF link unavailable)")

                                with st.expander("Show Abstract"):
                                    st.markdown(summary)
                        st.markdown("---") # End divider after results
                        st.success(f"Displayed {len(results)} results.", icon="✅")

                except Exception as e:
                    st.error(f"Error during arXiv search: {e}", icon="❌")
                    logger.error(f"arXiv Search Error: {e}", exc_info=True)
# --- End Search arXiv Tab ---


# --- Feedback Tab (Improved Presentation) ---
with tab_feedback:
    st.header("📝 Submit Feedback or Report Issues")
    st.markdown("""
    Your feedback is valuable for improving this tool! Please use the form below to share your thoughts,
    suggestions, or report any bugs you encounter.
    """)

    TALLY_ORIGINAL_EMBED_URL = "https://tally.so/embed/n0kkp6?alignLeft=1&hideTitle=1&transparentBackground=1&dynamicHeight=1"
    IFRAME_HEIGHT = 500 # Adjust height as needed based on form length
    logger.info(f"Embedding feedback form from Tally: {TALLY_ORIGINAL_EMBED_URL}")

    try:
        # Center the iframe or ensure it fits well
        components.iframe(TALLY_ORIGINAL_EMBED_URL, height=IFRAME_HEIGHT, scrolling=True)
        st.caption("Feedback form securely hosted by Tally.so.")
    except Exception as e:
        st.error(f"Could not load the feedback form.", icon="😞")
        st.markdown("You can also submit feedback directly [here](https://tally.so/r/n0kkp6).") # Provide direct link as fallback
        logger.error(f"Error embedding Tally iframe: {e}", exc_info=True)
# --- End Feedback Tab ---


# --- Footer (Slightly Enhanced) ---
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