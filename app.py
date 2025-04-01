# app.py
import streamlit as st
import os
import sys
import argparse # To mimic args for setup_data_and_fetch if needed
from typing import Optional, Tuple, List, Dict, Any
import traceback # For detailed error logging

# --- Path Setup ---
# Ensure this points to your project root correctly
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # Assumes app.py is in scripts/
# If app.py is in the root:
# project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"Project Root added to sys.path: {project_root}") # Debug print

# --- Import Refactored Logic ---
# Assuming these functions are refactored to be callable (no print/exit)
try:
    from hybrid_search_rag import config
    from scripts.cli import (
        load_components,
        check_nltk_data,
        run_recommendation,
        setup_data_and_fetch,
        run_arxiv_search,
        # We need access to the loaded components, assuming they are global in cli.py
        # This is not ideal, better if load_components returned them.
        # For now, we might need to import them if used directly, or trust load_components worked.
        # Example: from scripts.cli import loaded_metadata
    )
except ImportError as e:
    # Use st.exception for better error display in Streamlit
    st.exception(e)
    st.error(f"Failed to import project modules. Check sys.path and ensure refactoring is complete.")
    st.code(f"sys.path: {sys.path}")
    st.stop()
except Exception as e:
    st.exception(e)
    st.error("An unexpected error occurred during imports.")
    st.stop()

# --- Page Configuration (Wide Layout) ---
st.set_page_config(
    page_title="ML Study Recommender",
    page_icon="🧠", # Add a relevant emoji icon
    layout="wide"
)

# --- Title and Introduction ---
# Mimic the header style somewhat with title and markdown
st.title("🧠 ML Study Recommender")
st.markdown("""
Explore academic research effortlessly. This tool uses advanced AI (RAG & Hybrid Search)
to find relevant information, answer your questions from research papers, and provide cited summaries.
""")
st.divider() # Visual separator

# --- Caching Components ---
# Use st.cache_resource for expensive, non-serializable objects like models
@st.cache_resource
def cached_load_components():
    """Loads and caches the core RAG components."""
    print("Attempting to load components via Streamlit cache...")
    try:
        # Assuming load_components uses/populates globals in cli.py
        load_components(force_reload=False)
        # Indirect check (needs access to globals or return value)
        from scripts.cli import loaded_metadata # Example check
        if loaded_metadata is not None:
             print("Components loaded successfully (checked metadata).")
             return True
        else:
             print("Components seem not loaded after function call.")
             st.warning("Component loading function ran but components appear missing.")
             return False
    except Exception as e:
        st.error(f"Error loading RAG components:")
        st.exception(e) # Show full traceback in Streamlit UI
        print(f"Error loading RAG components: {traceback.format_exc()}")
        return False

@st.cache_data # Cache simple checks like NLTK data
def cached_check_nltk_data():
    """Checks and potentially downloads required NLTK data."""
    print("Checking NLTK data via Streamlit cache...")
    try:
        # Assuming check_nltk_data is refactored to not sys.exit()
        check_nltk_data()
        print("NLTK check complete.")
    except SystemExit:
         st.error("NLTK data download/verification failed. Check terminal logs.")
         st.stop()
    except Exception as e:
        st.error(f"Error during NLTK check:")
        st.exception(e)
        print(f"Error during NLTK check: {traceback.format_exc()}")
        st.stop()

# --- Initial Setup ---
# Run checks and component loading early
cached_check_nltk_data()
components_loaded = cached_load_components()

if not components_loaded:
    st.error("Core RAG components failed to load. Recommendation functionality will be disabled.")
    # Allow app to continue for other tabs like Fetch/Search

# --- UI Tabs ---
# Use relevant emojis for tabs
tab_rec, tab_how, tab_about, tab_fetch, tab_arxiv = st.tabs([
    "🧠 Recommend",
    "⚙️ How It Works",
    "ℹ️ About",
    "⏬ Fetch Data",
    "🔍 Search arXiv"
])

# --- Recommendation Tab ---
with tab_rec:
    st.header("Ask the RAG Assistant")
    st.markdown("Enter your topic or question below and get a cited answer based on the indexed documents.")

    # Use columns for better layout of controls
    col1_rec, col2_rec = st.columns([3, 1]) # Give more space to text area

    with col1_rec:
        query = st.text_area("Your Question:", value=config.DEFAULT_QUERY, height=150, key="rec_query", label_visibility="collapsed")

    with col2_rec:
        general_mode = st.checkbox("Hybrid Mode", value=False, key="rec_general", help="Allow LLM to use its general knowledge alongside retrieved documents.")
        concise_mode = st.checkbox("Concise Prompt", value=True, key="rec_concise", help="Use a more structured, concise prompt template for the LLM.")
        submit_rec = st.button("Get Recommendation", type="primary", key="rec_button", disabled=not components_loaded, use_container_width=True)

    if submit_rec:
        if not query:
            st.warning("Please enter a question.")
        else:
            mode_name = "Hybrid" if general_mode else "Strict"
            style_name = "Concise/Structured" if concise_mode else "Default/Detailed"
            st.info(f"Running recommendation ({mode_name} RAG, {style_name} Prompt)...")

            with st.spinner("Retrieving context and generating response..."):
                try:
                    # Call the backend function
                    answer, sources = run_recommendation(query, config.TOP_N_RESULTS, general_mode, concise_mode)

                    st.subheader("Assistant Response")
                    if answer:
                        # Use markdown to render formatting from the LLM response
                        st.markdown(answer, unsafe_allow_html=True) # Allow basic HTML if needed for citations/links
                    else:
                        st.warning("No response was generated by the LLM or fallback mechanism.")

                    # Put sources in an expander
                    with st.expander("Show Context Sources Provided to LLM"):
                        if sources:
                            # Display sources as a numbered list using markdown
                            source_markdown = "\n".join([f"{i+1}. {s.strip()}" for i, s in enumerate(sources)])
                            st.markdown(source_markdown)
                        else:
                            st.info("No relevant local document chunks were found or provided as context.")

                except Exception as e:
                    st.error(f"An error occurred during recommendation:")
                    st.exception(e) # Show traceback in UI
                    print(f"Recommendation Error: {traceback.format_exc()}")

# --- How It Works Tab ---
with tab_how:
    st.header("How the RAG System Works")
    st.markdown("This system uses a Retrieval-Augmented Generation (RAG) pipeline:")

    # Use columns to mimic the card layout from the HTML design
    col1_how, col2_how, col3_how = st.columns(3)

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
        * Citations `[N]` are automatically formatted.
        """)


    with col2_how:
        st.subheader("2. Hybrid Indexing")
        st.markdown("""
        * Generates dense vector embeddings (e.g., `BAAI/bge-large-en-v1.5`) for semantic understanding.
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
with tab_about:
    st.header("About This Project")

    # Use columns for better layout
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
        LLM integration (Gemini), and dynamic source finding are implemented.
        """)

        st.subheader("Developer")
        st.markdown(f"""
        This project is developed by **Felix Nathaniel**, a Computer Science student at BINUS University,
        as part of ongoing learning and exploration in AI/ML.
        *(Powered by Streamlit)*
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
with tab_fetch:
    st.header("Update Knowledge Base")
    st.warning("Fetching and processing can take significant time and requires internet access.", icon="⏳")

    # Use a form to group inputs for the fetch operation
    with st.form("fetch_form"):
        st.subheader("Source Selection")
        fetch_method = st.radio(
            "Select Fetch Method:",
            ("Use Defaults (from config.py)",
             "Use Custom arXiv Query",
             "Use LLM Suggestion (Requires Topic)"),
            key="fetch_method",
            horizontal=True # Make radio buttons horizontal
        )

        # Layout inputs using columns
        col1_fetch, col2_fetch = st.columns(2)
        with col1_fetch:
            custom_arxiv_query = st.text_input("Custom arXiv Query:", key="fetch_arxiv_query", help="Used if 'Custom arXiv Query' is selected.")
            topic = st.text_input("Topic for LLM Suggestion:", key="fetch_topic", help="Used if 'LLM Suggestion' is selected.")
        with col2_fetch:
            num_arxiv_results = st.number_input(f"Max arXiv Results:", min_value=5, max_value=500, value=config.MAX_ARXIV_RESULTS, key="fetch_num_arxiv", help="Applies to default, custom, or suggested query.")

        st.divider()
        submitted_fetch = st.form_submit_button("Fetch and Process Data", type="primary", use_container_width=True)

        if submitted_fetch:
            # Prepare arguments namespace for setup_data_and_fetch
            fetch_args = argparse.Namespace()
            fetch_args.suggest_sources = (fetch_method == "Use LLM Suggestion (Requires Topic)")
            fetch_args.topic = topic if fetch_args.suggest_sources else None
            fetch_args.arxiv_query = custom_arxiv_query if fetch_method == "Use Custom arXiv Query" else None
            fetch_args.num_arxiv = num_arxiv_results

            # Input Validation
            valid_input = True
            if fetch_args.suggest_sources and not fetch_args.topic:
                st.error("Topic is required when using LLM Suggestion.")
                valid_input = False
            if fetch_method == "Use Custom arXiv Query" and not fetch_args.arxiv_query:
                 st.error("Custom arXiv Query is required for that fetch method.")
                 valid_input = False

            if valid_input:
                st.info("Starting data fetch and processing... See terminal for detailed logs.")
                with st.spinner("Fetching sources, chunking, embedding, indexing... Please wait."):
                    try:
                        # Call the backend function
                        status_message = setup_data_and_fetch(fetch_args)

                        # Display result and handle component reload
                        if status_message.lower().startswith("success"):
                            st.success(f"Fetch Successful: {status_message}")
                            st.info("Clearing component cache and reloading...")
                            # --- IMPORTANT: Reload components ---
                            cached_load_components.clear() # Clear the cache
                            # Rerun the script to force reload and reflect changes
                            st.rerun()
                        elif status_message.lower().startswith("error"):
                            st.error(f"Fetch Failed: {status_message}")
                        else:
                            st.warning(f"Fetch Status: {status_message}")

                    except Exception as e:
                        st.error(f"An critical error occurred during data fetching/processing:")
                        st.exception(e)
                        print(f"Fetch Error: {traceback.format_exc()}")

# --- Search arXiv Tab ---
with tab_arxiv:
    st.header("Direct arXiv Search")
    st.markdown("Perform a live search directly on arXiv.org.")

    # Layout inputs using columns
    col1_arxiv, col2_arxiv = st.columns([3, 1])
    with col1_arxiv:
        arxiv_query = st.text_input("Search query for arXiv:", key="arxiv_query_input", label_visibility="collapsed", placeholder="Enter arXiv search query...")
    with col2_arxiv:
        num_results = st.number_input("Max Results:", min_value=1, max_value=100, value=10, key="arxiv_num_input", label_visibility="collapsed")

    if st.button("Search arXiv", key="arxiv_search_exec_button", type="primary", use_container_width=True):
        if not arxiv_query:
            st.warning("Please enter an arXiv search query.")
        else:
            st.info(f"Searching arXiv for '{arxiv_query}'...")
            with st.spinner("Searching arXiv..."):
                try:
                    # Call the backend function
                    results = run_arxiv_search(arxiv_query, num_results)

                    if not results:
                        st.info("No results found on arXiv for this query.")
                    else:
                        st.subheader(f"Found {len(results)} results:")
                        # Display results more cleanly
                        for i, paper in enumerate(results):
                            st.markdown(f"**{i + 1}. {paper.get('title', 'N/A')}**")
                            st.caption(f"Authors: {', '.join(paper.get('authors', [])) or 'N/A'} | Published: {paper.get('published', 'N/A')}")
                            pdf_url = paper.get('pdf_url')
                            if pdf_url and pdf_url != '#':
                                st.link_button("View PDF", pdf_url) # Use link_button
                            with st.expander("Show Abstract"):
                                st.write(paper.get('summary', 'N/A'))
                            st.divider() # Visual separator

                except Exception as e:
                    st.error(f"An error occurred during arXiv search:")
                    st.exception(e)
                    print(f"arXiv Search Error: {traceback.format_exc()}")

# --- Footer ---
st.divider()
st.markdown("<div style='text-align: center; color: grey;'>© 2025 Felix Nathaniel | ML Study Recommender</div>", unsafe_allow_html=True)

