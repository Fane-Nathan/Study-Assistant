# app.py
import streamlit as st
import nltk
import ssl
import os
import sys
import argparse
import traceback
# Removed: import requests (if not used elsewhere)
import streamlit.components.v1 as components
# Removed: import multiprocessing
# Removed: import time
import logging # Use logging for better control
# Removed: import socket

# --- Basic Logging Setup for Streamlit App ---
# Gotta see what's happening! Setting up basic logging to catch INFO level messages.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [App] %(message)s', force=True)
logger = logging.getLogger(__name__)

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
# Making the Streamlit page look fancy! Title, icon, wide layout - the works.
st.set_page_config(
    page_title="Your Study Assistant", # <--- UPDATED TITLE
    page_icon="📚",
    layout="wide"
)
# --- End Page Configuration ---


# --- Remove Proxy Server Import and Startup Logic ---
# Yeah, we ditched the whole proxy thing. Too much hassle!
# Direct embedding of the Tally form is WAY easier. KISS principle, right? 😉
logger.info("Proxy server logic has been removed. Embedding Tally form directly.")
# --- End Proxy Removal ---


# --- NLTK Data Download on Startup ---
# This part is kinda funky but super important. It checks if NLTK (a text processing library)
# has the 'punkt' (for splitting sentences) and 'stopwords' (common words like 'the', 'a') data it needs.
# If not, it tries to download them. Sometimes SSL certificates cause drama, so there's a workaround here.
# Without this data, the app might just throw a tantrum later. 💥
try:
    # This is like telling Python, "Hey, chill with the SSL certificate checks for downloading, okay?"
    # It's a bit of a hack for environments where default checks might fail. ONLY FOR NLTK DOWNLOAD!
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # If this weird SSL context thing doesn't exist, no worries, just move on.
    pass
else:
    # If it DOES exist, replace the default way Python handles secure connections with the "chill" way.
    ssl._create_default_https_context = _create_unverified_https_context

NLTK_PACKAGES = ['punkt', 'stopwords']
download_needed_flag = False
# We use session_state so Streamlit doesn't keep checking/downloading NLTK stuff EVERY SINGLE TIME the script reruns (which it does a lot!).
if 'nltk_data_checked' not in st.session_state:
    st.session_state.nltk_data_checked = False

if not st.session_state.nltk_data_checked:
    logger.info("Checking NLTK data...")
    # Show a little spinner so the user knows we're doing something important...zzzzz...
    with st.spinner("Checking/downloading NLTK data..."):
        all_packages_found = True
        for package in NLTK_PACKAGES:
            try:
                # Try finding the package. This is like asking NLTK, "Yo, you got this?"
                nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
                logger.info(f"NLTK package '{package}' found.")
            except LookupError:
                # Oops, NLTK doesn't have it! Gotta download.
                logger.warning(f"NLTK package '{package}' not found. Triggering download.")
                download_needed_flag = True
                try:
                    # Go fetch! Using quiet=True so it doesn't print too much junk.
                    nltk.download(package, quiet=True)
                    logger.info(f"NLTK '{package}' download attempt finished.")
                    # Double-check if it's really there now. Like, really really there?
                    nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
                    logger.info(f"NLTK '{package}' found after download attempt.")
                except Exception as e:
                    # Oh no, download failed! This is bad. App can't work without this. Panic! 😱
                    st.error(f"Fatal Error: Failed to download required NLTK package '{package}'. App cannot continue. Error: {e}")
                    logger.error(f"NLTK '{package}' download failed: {e}", exc_info=True)
                    all_packages_found = False
                    st.stop() # Stop the app. Game over, man.
            except Exception as E_find:
                 # Something else went wrong just checking for the package. Also bad.
                 st.error(f"Fatal Error checking for NLTK package '{package}'. App cannot continue. Error: {E_find}")
                 logger.error(f"NLTK '{package}' check failed: {E_find}", exc_info=True)
                 all_packages_found = False
                 st.stop() # Seriously, stop.

        if download_needed_flag: logger.info("NLTK download process completed.")
        if all_packages_found:
            # Phew! Got everything. Mark it as checked so we don't do this again this session.
            st.session_state.nltk_data_checked = True
            logger.info("NLTK session state marked as checked.")
        else:
             # If we somehow got here without stopping but things failed... warn the user.
             logger.error("Not all NLTK packages were found/downloaded successfully.")
             st.error("Failed to prepare all required NLTK packages. App may not function correctly.")
# --- End NLTK Data Download ---

# --- Path Setup ---
# This makes sure Python can find our other project files (like config.py, recommender.py etc.)
# It adds the main project folder to the list of places Python looks for code. Super handy!
logger.info("Setting up sys.path...")
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
logger.info(f"Project Root added to sys.path: {project_root}")
# --- End Path Setup ---


# --- DELAYED Import Refactored Logic ---
# We try to import our project's own code modules here.
# Doing it after the path setup ensures Python *can* find them.
# If this fails, the app is basically useless, so we set a flag. 🚩
logger.info("Attempting project module imports...")
project_modules_loaded = False # Default to False
try:
    # Pulling in the config settings, and the main functions from our command-line script (cli.py).
    # Reusing code is good, mmmkay?
    from hybrid_search_rag import config
    from scripts.cli import ( load_components, run_recommendation,
                              setup_data_and_fetch, run_arxiv_search )
    project_modules_loaded = True # Yay, it worked!
    logger.info("Project module imports successful.")
except ImportError as e:
    # Uh oh, couldn't import something. Maybe a typo? Or the path setup failed?
    # Show an error in the app and log the details. Functionality will be limited.
    st.error(f"Failed to import project modules (ImportError). Check setup. Error: {e}")
    st.code(f"sys.path: {sys.path}") # Show the paths Python checked.
    logger.error(f"Project import failed (ImportError): {e}", exc_info=True)
except Exception as e:
    # Some other random error happened during import. Still bad news.
    st.error(f"An unexpected error occurred during project imports: {e}. App limited.")
    logger.error(f"Project import failed (Exception): {e}", exc_info=True)
# --- End DELAYED Import Refactored Logic ---


# --- Title and Introduction ---
# Just setting the title and adding some introductory text. Pretty standard stuff.
st.title("📚 Research Paper Assistant") # <--- UPDATED TITLE
st.markdown("""
Explore academic research effortlessly. This tool uses advanced AI (RAG & Hybrid Search)
to find relevant information, answer your questions from research papers, and provide cited summaries.
""")

# --- *** ADD DISCLAIMER HERE *** ---
st.info("""
**Disclaimer:** This tool is designed to assist in finding and ranking relevant research papers based on your queries.
It is **not** intended as a replacement for general-purpose AI chatbots (like ChatGPT, Gemini, etc.) for conversational tasks or broad knowledge questions.
""", icon="ℹ️")
# --- *** END DISCLAIMER *** ---

st.divider() # Adds a nice little line separator.

# --- Caching Components ---
# This is a Streamlit superpower! @st.cache_resource tells Streamlit to load these components
# (like the big embedding model and search indexes) only ONCE and then reuse them.
# Saves a TON of time and computation, especially when the user interacts with the app.
# Without this, loading would happen on every single click. Yikes! 😬
@st.cache_resource
def cached_load_components():
    """Loads and caches the core RAG components."""
    logger.info("Attempting to load components via cached_load_components...")
    if not project_modules_loaded:
        # If the earlier imports failed, we can't load components. Bail out.
        logger.error("Cannot load components because project modules failed to import.")
        return False
    try:
        # Call the actual loading function from cli.py. force_reload=False means use cached data if available.
        load_components(force_reload=False)
        # Check if the 'loaded_metadata' variable (which load_components should set) actually exists now.
        # This is a slightly indirect way to confirm loading worked.
        from scripts.cli import loaded_metadata # Import it here AFTER calling load_components
        if loaded_metadata is not None:
             logger.info("Components loaded successfully (checked indicator).")
             return True # Success!
        else:
             # Hmm, the function ran but the data seems missing? Weird. Warn the user.
             logger.warning("Component loading function ran but components appear missing (indicator is None).")
             st.warning("Component loading function ran but components appear missing.")
             return False
    except NameError as ne:
         # This usually means 'loaded_metadata' wasn't found after calling load_components. Problem!
         st.error(f"Error loading RAG components: Likely a problem accessing state from 'scripts.cli'. Did load_components run correctly? Error: {ne}")
         logger.error(f"NameError during component loading: {ne}", exc_info=True)
         return False
    except Exception as e:
        # Catch-all for other loading errors.
        st.error(f"Error loading RAG components: {e}")
        logger.error(f"Exception during component loading: {e}", exc_info=True)
        return False

# --- Initial Setup ---
# Actually call the cached loading function now.
logger.info("Calling cached_load_components...")
# Only try to load if the project modules were imported successfully earlier.
components_load_status = cached_load_components() if project_modules_loaded else False
logger.info(f"Components loaded status: {components_load_status}")
components_loaded = components_load_status # Store the status
if not components_loaded and project_modules_loaded:
    # If modules loaded BUT components didn't, show an error. Recommendations won't work.
    st.error("Core RAG components failed to load. Recommendation functionality will be disabled.")
elif not project_modules_loaded:
     # If project modules didn't load, we already showed an error. No need for another one here.
     pass

# --- UI Tabs ---
# Setting up the different tabs the user can click on in the Streamlit app.
logger.info("Defining UI tabs...")
tab_rec, tab_how, tab_about, tab_fetch, tab_arxiv, tab_feedback = st.tabs([
    "🧠 Recommend",
    "⚙️ How It Works",
    "ℹ️ About",
    "⏬ Fetch Data",
    "🔍 Search arXiv",
    "📝 Feedback"
])
logger.info("UI tabs defined.")

# --- Recommendation Tab ---
# This 'with' block defines what shows up in the 'Recommend' tab.
with tab_rec:
    st.header("Ask the RAG Assistant")
    st.markdown("Enter your topic or question below and get a cited answer based on the indexed documents.")
    # Use columns to arrange the text area and checkboxes side-by-side.
    col1_rec, col2_rec = st.columns([3, 1]) # Column 1 is 3x wider than column 2.
    with col1_rec:
        # Set a default query text. If our project modules loaded, use the one from config, otherwise use a generic example.
        default_query = config.DEFAULT_QUERY if project_modules_loaded else "Example: What is Retrieval-Augmented Generation?"
        # The main text area for the user's question.
        # Use session_state to remember the last query if the app reruns
        query = st.text_area("Your Question:", value=st.session_state.get("rec_query", default_query), height=150, key="rec_query_input_area", label_visibility="collapsed")
    with col2_rec:
        # Checkbox for enabling "Hybrid Mode" (LLM uses general knowledge too).
        # Use session_state to remember the last mode selection.
        general_mode = st.checkbox("Hybrid Mode", value=st.session_state.get("rec_general", False), key="rec_general_toggle", help="Allow LLM to use its general knowledge alongside retrieved documents.")
        # Checkbox for using a more concise prompt for the LLM.
        # Use session_state to remember the last mode selection.
        concise_mode = st.checkbox("Concise Prompt", value=st.session_state.get("rec_concise", True), key="rec_concise_toggle", help="Use a more structured, concise prompt template for the LLM.")
        # The main button to submit the query. It's disabled if the core components didn't load.
        submit_rec = st.button("Get Recommendation", type="primary", key="rec_button", disabled=not components_loaded, use_container_width=True)

    # This code runs ONLY when the user clicks the 'Get Recommendation' button.
    if submit_rec:
        if not query:
            # Gotta enter *something*!
            st.warning("Please enter a question.")
        elif not components_loaded:
            # Can't recommend if the engine isn't running!
            st.error("Cannot get recommendation: Core components failed to load.")
        else:
            # Okay, let's do this!
            # --- Store inputs in session_state before running ---
            st.session_state.rec_query = query
            st.session_state.rec_general = general_mode
            st.session_state.rec_concise = concise_mode
            # ---
            mode_name = "Hybrid" if general_mode else "Strict"
            style_name = "Concise/Structured" if concise_mode else "Default/Detailed"
            st.info(f"Running recommendation ({mode_name} RAG, {style_name} Prompt)...")
            # Show a spinner while we work... it can take a few seconds (or more!). ⏳
            with st.spinner("Retrieving context and generating response..."):
                try:
                    # How many results to aim for? Get from config if possible.
                    top_n = config.TOP_N_RESULTS if project_modules_loaded else 5
                    # Call the actual recommendation function (from cli.py). This does the heavy lifting!
                    # It now returns two values: the LLM's response string and the list of context source strings.
                    llm_answer, context_sources = run_recommendation(
                        query=query,
                        num_final_results=top_n, # Use top_n for fallback display if needed
                        general_mode=general_mode,
                        concise_mode=concise_mode
                        )
                    # --- Store results in session_state ---
                    # We store the results so they persist even if the user interacts with something else causing a rerun.
                    st.session_state.llm_answer = llm_answer
                    st.session_state.context_sources = context_sources
                    # ---

                except Exception as e:
                    # Catch any errors during the recommendation process itself.
                    st.error(f"An error occurred during recommendation: {e}")
                    logger.error(f"Recommendation Error: {e}", exc_info=True)
                    # Clear previous results on error
                    st.session_state.llm_answer = None
                    st.session_state.context_sources = []

    # --- Display the results (use stored results from session state) ---
    # This block runs *after* the button is clicked and results (or errors) are stored,
    # or on subsequent reruns if results already exist in session_state.

    # Check if we have a stored answer to display.
    if "llm_answer" in st.session_state and st.session_state.llm_answer:
        st.markdown("---") # Separator line
        st.subheader("Assistant Response")
        # Display the main LLM answer. Because we modified the prompts in cli.py,
        # this answer should now *include* the "## Citations:" section if the LLM behaved.
        st.markdown(st.session_state.llm_answer, unsafe_allow_html=True) # unsafe_allow_html helps render Markdown formatting.

    # Check if we have stored context sources to display.
    if "context_sources" in st.session_state: # Check if key exists
         st.markdown("---") # Separator line
         # Use an expander - a collapsible section.
         with st.expander("Show Context Sources Provided to LLM"):
             sources = st.session_state.context_sources
             if sources: # Check if list is not empty
                 # Join with single newline for tighter list format
                formatted_sources = "  \n".join(sources)
                st.markdown(formatted_sources)
             else: # Handle case where list exists but is empty
                 st.info("No relevant local document chunks were provided as context for this answer.")
    # else: # If key doesn't exist yet (e.g., first run before any search) - optional
    #     pass # Don't display the expander yet
    # --- End display logic ---


# --- How It Works Tab ---
# Just some static text explaining the RAG pipeline. Good for users who are curious.
# Uses columns again for better layout.
with tab_how:
    st.header("How the RAG System Works")
    st.markdown("This system uses a Retrieval-Augmented Generation (RAG) pipeline:")
    col1_how, col2_how, col3_how = st.columns(3)
    # Try to show the actual embedding model name from config if possible.
    emb_model = config.EMBEDDING_MODEL_NAME if project_modules_loaded else "an embedding model"
    with col1_how:
        st.subheader("1. Data Ingestion & Processing")
        st.markdown("""
        * Fetches data from arXiv, web URLs, and PDFs (using PyMuPDF).
        * Cleans text and applies sentence-based chunking.
        * Breaks down long documents for better analysis.
        """)
        st.subheader("4. LLM Generation & Citation")
        st.markdown("""
        * Retrieved chunks are passed to a large language model (e.g., Gemini).
        * LLM generates a coherent answer based on context (Strict RAG) or context + general knowledge (Hybrid Mode).
        * Citations `[N]` are automatically formatted based on instructions.
        """)
    with col2_how:
        st.subheader("2. Hybrid Indexing")
        # Use an f-string to insert the embedding model name dynamically. Cool, huh? 😎
        st.markdown(f"""
        * Generates dense vector embeddings (using `{emb_model}`) for semantic understanding.
        * Builds a sparse keyword index (BM25) for term matching.
        * Indexes are persisted locally.
        """)
        st.subheader("5. RAG Pipeline")
        st.markdown("""
        * The entire process forms a RAG pipeline.
        * Enhances LLM responses with specific, retrieved information.
        * Improves factual grounding and reduces hallucinations.
        """)
    with col3_how:
        st.subheader("3. Hybrid Retrieval (RRF)")
        st.markdown("""
        * Queries both embedding and BM25 indexes simultaneously.
        * Combines results using Reciprocal Rank Fusion (RRF).
        * Produces a single, relevance-ranked list of document chunks.
        """)
        st.subheader("6. Core Implementation")
        st.markdown("""
        * Built using Python with modular components.
        * Features include dynamic source finding, chunking, hybrid search, and configurable LLM integration.
        * This Streamlit app provides the UI.
        """)
    st.divider()

# --- About Tab ---
# More static info, this time about the project itself.
with tab_about:
    st.header("About This Project")
    col1_about, col2_about = st.columns(2)
    with col1_about:
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
        st.subheader("Developer")
        # Added my name here since I'm working on it! Feel free to personalize further.
        st.markdown(f"""
        This project is developed by **Felix Nathaniel**, **Reynaldi Anatyo**, & **Dennison Soebdibjo** a Computer Science student at BINUS University,
        as part of ongoing learning and exploration in AI/ML.
        """)
    with col2_about:
        st.subheader("Future Work")
        st.markdown("""
        * **Expand Corpus:** Build a larger, diverse, curated knowledge base.
        * **Tune Retrieval:** Systematically evaluate and optimize retrieval performance.
        * **Implement Recommender Features:** Add proactive topic suggestions.
        * **Add Personalization:** Incorporate user profiles, history, etc.
        * **Refine UI/UX:** Continuously improve the web interface.
        * **(Maybe) API Endpoint:** Create a dedicated API for broader integration.
        """)
    st.divider()


# --- Fetch Data Tab ---
# This tab allows the user to trigger a data re-fetch and processing.
with tab_fetch:
    st.header("Update Knowledge Base")
    # Important warning! Fetching can take ages and needs internet. ⏳
    st.warning("Fetching and processing can take significant time and requires internet access.", icon="⏳")
    # Disable the whole form if the project modules didn't load earlier.
    fetch_disabled = not project_modules_loaded
    if fetch_disabled:
        st.warning("Data fetching is disabled because core project modules failed to load.")

    # Use a Streamlit form to group inputs and have a single submit button.
    with st.form("fetch_form"):
        st.subheader("Source Selection")
        # Radio buttons to choose the fetch method.
        fetch_method = st.radio(
            "Select Fetch Method:",
            ("Use Defaults (from config.py)",
            "Use Custom arXiv Query",
            "Use LLM Suggestion (Requires Topic)"),
            key="fetch_method",
            horizontal=True, # Make them appear side-by-side
            disabled=fetch_disabled
        )
        col1_fetch, col2_fetch = st.columns(2) # Columns for inputs
        with col1_fetch:
            # Input for custom arXiv query (only used if that method is selected).
            custom_arxiv_query = st.text_input("Custom arXiv Query:", key="fetch_arxiv_query", help="Used if 'Custom arXiv Query' is selected.", disabled=fetch_disabled)
            # Input for the topic if using LLM suggestions.
            topic = st.text_input("Topic for LLM Suggestion:", key="fetch_topic", help="Used if 'LLM Suggestion' is selected.", disabled=fetch_disabled)
        with col2_fetch:
            # Number input for max arXiv results. Default from config if possible.
            default_arxiv_results = config.MAX_ARXIV_RESULTS if project_modules_loaded else 10
            num_arxiv_results = st.number_input(f"Max arXiv Results:", min_value=5, max_value=500, value=default_arxiv_results, key="fetch_num_arxiv", help="Applies to default, custom, or suggested query.", disabled=fetch_disabled)
        st.divider()
        # The big red button... well, primary blue button.
        submitted_fetch = st.form_submit_button("Fetch and Process Data", type="primary", use_container_width=True, disabled=fetch_disabled)

        # This code runs ONLY when the 'Fetch and Process Data' button inside the form is clicked.
        if submitted_fetch and not fetch_disabled:
            # Create a simple 'Namespace' object to mimic the 'args' object from argparse.
            # This lets us reuse the setup_data_and_fetch function from cli.py easily. Clever! ✨
            fetch_args = argparse.Namespace()
            fetch_args.suggest_sources = (fetch_method == "Use LLM Suggestion (Requires Topic)")
            fetch_args.topic = topic if fetch_args.suggest_sources else None
            fetch_args.arxiv_query = custom_arxiv_query if fetch_method == "Use Custom arXiv Query" else None
            fetch_args.num_arxiv = num_arxiv_results
            valid_input = True # Assume input is valid initially
            # Basic input validation
            if fetch_args.suggest_sources and not fetch_args.topic:
                st.error("Topic is required when using LLM Suggestion.")
                valid_input = False
            if fetch_method == "Use Custom arXiv Query" and not fetch_args.arxiv_query:
                st.error("Custom arXiv Query is required for that fetch method.")
                valid_input = False

            if valid_input:
                # Okay, inputs look good. Let's fetch!
                st.info("Starting data fetch and processing... See terminal for detailed logs.")
                # Another spinner for the long wait...
                with st.spinner("Fetching sources, chunking, embedding, indexing... Please wait."):
                    try:
                        # Call the fetch function from cli.py with our constructed args.
                        status_message = setup_data_and_fetch(fetch_args)
                        # Check the message returned by the function.
                        if status_message.lower().startswith("success"):
                            st.success(f"Fetch Successful: {status_message}")
                            # IMPORTANT: If fetch worked, clear the component cache!
                            # This forces the app to reload the data next time it needs it.
                            st.info("Clearing component cache to reload new data...")
                            cached_load_components.clear() # Zap the cache! 💥
                            st.rerun() # Force Streamlit to rerun the *entire* script to reload components.
                        elif status_message.lower().startswith("error"):
                            st.error(f"Fetch Failed: {status_message}")
                        else:
                            # Some other status (like a warning maybe)
                            st.warning(f"Fetch Status: {status_message}")
                    except Exception as e:
                        # Catch critical errors during the fetch process itself.
                        st.error(f"An critical error occurred during data fetching/processing: {e}")
                        logger.error(f"Fetch Error: {e}", exc_info=True)

# --- Search arXiv Tab ---
# Lets the user do a direct, live search on arXiv without affecting the app's indexed data.
with tab_arxiv:
    st.header("Direct arXiv Search")
    st.markdown("Perform a live search directly on arXiv.org.")
    # Disable if project modules failed to load (need run_arxiv_search function).
    search_disabled = not project_modules_loaded
    if search_disabled:
        st.warning("Direct arXiv search is disabled because core project modules failed to load.")

    col1_arxiv, col2_arxiv = st.columns([3, 1])
    with col1_arxiv:
        # Input for the search query.
        arxiv_query = st.text_input("Search query for arXiv:", key="arxiv_query_input", label_visibility="collapsed", placeholder="Enter arXiv search query...", disabled=search_disabled)
    with col2_arxiv:
        # Number input for how many results to show.
        num_results = st.number_input("Max Results:", min_value=1, max_value=100, value=10, key="arxiv_num_input", label_visibility="collapsed", disabled=search_disabled)

    # The search button.
    if st.button("Search arXiv", key="arxiv_search_exec_button", type="primary", use_container_width=True, disabled=search_disabled):
        if not arxiv_query:
            # Need a query!
            st.warning("Please enter an arXiv search query.")
        else:
            st.info(f"Searching arXiv for '{arxiv_query}'...")
            # Spinner while searching...
            with st.spinner("Searching arXiv..."):
                try:
                    # Call the search function from cli.py.
                    results = run_arxiv_search(arxiv_query, num_results)
                    if not results:
                        st.info("No results found on arXiv for this query.")
                    else:
                        st.subheader(f"Found {len(results)} results:")
                        # Loop through the results and display them nicely.
                        for i, paper in enumerate(results):
                            st.markdown(f"**{i + 1}. {paper.get('title', 'N/A')}**")
                            st.caption(f"Authors: {', '.join(paper.get('authors', [])) or 'N/A'} | Published: {paper.get('published', 'N/A')}")
                            pdf_url = paper.get('pdf_url')
                            # Make sure the PDF URL is actually a valid URL before making a button.
                            if isinstance(pdf_url, str) and pdf_url.startswith('http'):
                                st.link_button("View PDF", pdf_url) # Handy button to open the PDF.
                            else:
                                # If the URL is weird or missing, log it.
                                logger.warning(f"Invalid or missing PDF URL for paper '{paper.get('title', 'N/A')}': {pdf_url}")
                            # Put the abstract in an expander so it doesn't clutter the page.
                            with st.expander("Show Abstract"):
                                st.write(paper.get('summary', 'N/A'))
                            st.divider() # Separator between papers.
                except Exception as e:
                    # Catch errors during the arXiv search itself.
                    st.error(f"An error occurred during arXiv search: {e}")
                    logger.error(f"arXiv Search Error: {e}", exc_info=True)


# ***** START: Feedback Tab Section using DIRECT iframe embedding *****
# This tab embeds the Tally feedback form directly using an iframe.
# Simpler than the old proxy method! 👍
with tab_feedback:
    st.header("Submit Feedback")

    # The URL provided by Tally for embedding.
    TALLY_ORIGINAL_EMBED_URL = "https://tally.so/embed/n0kkp6?alignLeft=1&hideTitle=1&transparentBackground=1&dynamicHeight=1"

    # Set a height for the iframe. Might need tweaking if the form changes.
    IFRAME_HEIGHT = 500 # Increased height slightly, adjust if needed

    logger.info(f"Embedding feedback form directly from Tally: {TALLY_ORIGINAL_EMBED_URL}")
    try:
        # Use Streamlit's built-in iframe component. Easy peasy!
        components.iframe(TALLY_ORIGINAL_EMBED_URL, height=IFRAME_HEIGHT, scrolling=True)
        st.caption("Feedback form provided by Tally.so.") # Give credit where it's due.
    except Exception as e:
        # Just in case embedding fails somehow.
        st.error(f"Failed to load the feedback form iframe directly from Tally.")
        st.error(f"Error details: {e}")
        logger.error(f"Error embedding direct Tally iframe: {e}", exc_info=True)
# ***** END: Feedback Tab Section *****


# --- Footer ---
# A nice little footer at the bottom of the page.
st.divider()
st.markdown("<div style='text-align: center; color: grey;'>© 2025 Felix Nathaniel, Reynaldi Anatyo, Dennison Soebdibjo | Research Paper Assistant</div>", unsafe_allow_html=True) #<-- UPDATED FOOTER TITLE