# StudyAssistant: Hybrid Search RAG CLI Tool

StudyAssistant is a command-line tool designed to assist with research and learning by fetching information from arXiv and the web, processing it using advanced NLP techniques (chunking, embeddings, BM25), and providing answers or summaries through Retrieval-Augmented Generation (RAG) powered by Large Language Models (LLMs). It features hybrid search (semantic + keyword) and configurable RAG modes.

## Features

* **Data Fetching:**
    * Retrieve papers from arXiv based on search queries.
    * Crawl and fetch articles from web URLs, handling dynamic JavaScript-rendered content using Playwright.
    * LLM-powered source suggestion for targeted data fetching based on a topic.
* **Data Processing:**
    * Chunk documents into manageable pieces based on sentence boundaries.
    * Generate high-quality text embeddings using Google's Gemini models (configurable).
    * Build BM25 keyword indexes for efficient sparse retrieval.
    * Store processed data (metadata, embeddings, index) locally.
* **Hybrid Retrieval:**
    * Combine semantic search (vector embeddings) and keyword search (BM25) results using rank fusion for robust relevance ranking.
* **Retrieval-Augmented Generation (RAG):**
    * Answer user queries by feeding retrieved document chunks as context to a powerful LLM (e.g., Gemini).
    * **Strict RAG Mode:** Generate answers based *only* on the information found in the fetched local documents.
    * **Hybrid RAG Mode:** Allow the LLM to use both the retrieved local documents *and* its general knowledge base for more comprehensive answers.
    * **Output Formatting:** Supports detailed, conversational responses or concise, structured summaries.
* **Direct arXiv Search:** Search arXiv directly without processing or RAG.
* **Command-Line Interface:** Easy-to-use CLI for interacting with all features.
* **Configurable:** Key settings (API keys, model names, paths, parameters) managed via `config.py`.

## Installation

1.  **Clone the Repository:**
    ```bash
    # Replace <repo_url> with the actual URL if you have one
    # git clone <repo_url>
    cd StudyAssistant
    ```

2.  **Set Up Python Environment:**
    It's highly recommended to use a virtual environment (like Conda or venv). Python 3.9+ is recommended.
    ```bash
    # Example using Conda
    conda create -n study_env python=3.9 -y
    conda activate study_env
    ```

3.  **Install Python Packages:**
    Create a `requirements.txt` file in the project root with the following content:
    ```txt
    # requirements.txt
    playwright
    readability-lxml
    numpy
    rank_bm25
    nltk
    arxiv
    requests
    PyMuPDF
    beautifulsoup4
    google-generativeai # Or other LLM provider SDK
    # Add python-dotenv if using .env for API keys
    # python-dotenv
    ```
    Then install the packages:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install NLTK Data:**
    The script attempts to download required NLTK data (`punkt` for sentence tokenization, `stopwords`) automatically on first run if missing. You can also install it manually:
    ```bash
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
    ```

5.  **Install Playwright Browsers & Dependencies:**
    Playwright needs browser binaries and system dependencies.
    ```bash
    # Install browser binaries (Chromium, Firefox, WebKit)
    playwright install

    # Install OS dependencies (REQUIRED on Linux)
    # This command detects your OS (like Fedora) and installs needed libs.
    sudo playwright install-deps
    ```
    *(If `sudo playwright install-deps` fails, refer to Playwright documentation for manual installation instructions for your specific Linux distribution).*

## Configuration

Configuration is crucial, especially for API keys and model selection.

1.  **Locate/Create `config.py`:** Find or create the configuration file at `hybrid_search_rag/config.py`.
2.  **Edit Settings:** Modify the settings within `config.py`. Key settings include:
    * `GOOGLE_API_KEY`: **Required** for Gemini embedding and LLM calls. **Do NOT commit your API key directly!** Consider using environment variables or a `.env` file (see below).
    * `EMBEDDING_MODEL_NAME`: e.g., `'models/embedding-001'`
    * `LLM_PROVIDER`: e.g., `'gemini'` (used conceptually in logging/prompts)
    * `LLM_MODEL_ID`: e.g., `'gemini-1.5-flash-latest'` or `'gemini-pro'`
    * `DATA_DIR`: Path to store processed data (e.g., `"data/"`). Ensure this directory exists.
    * `METADATA_FILE`, `EMBEDDINGS_FILE`, `BM25_INDEX_FILE`: Filenames within `DATA_DIR`.
    * `TARGET_WEB_URLS`: Default list of URLs to crawl in `Workspace`.
    * `DEFAULT_ARXIV_QUERY`: Default query for `Workspace`.
    * Crawler/RAG parameters (`MAX_PAGES_TO_CRAWL`, `RAG_NUM_DOCS`, etc.)

3.  **API Key Security (Recommended):**
    Instead of hardcoding the API key in `config.py`, use environment variables.
    * **Option A: Environment Variable:**
        Set the variable in your terminal before running:
        ```bash
        export GOOGLE_API_KEY='YOUR_ACTUAL_API_KEY'
        python scripts/cli.py ...
        ```
        Modify `config.py` to read it:
        ```python
        # config.py
        import os
        from dotenv import load_dotenv # If using .env

        load_dotenv() # Load variables from .env file if it exists

        GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        # ... rest of config
        ```
    * **Option B: `.env` file:**
        Install `python-dotenv` (`pip install python-dotenv`). Create a file named `.env` in the project root:
        ```.env
        GOOGLE_API_KEY=YOUR_ACTUAL_API_KEY
        ```
        **Add `.env` to your `.gitignore` file** to prevent committing secrets. The `load_dotenv()` call (shown above) in `config.py` will load it.

## Usage

Run all commands from the project's root directory (`StudyAssistant/`).

**1. Fetch and Process Data (`Workspace`)**

* **Default Fetch (Uses `config.py` defaults):** Fetches default arXiv query and web URLs, processes, and saves data. Required before using `recommend`.
    ```bash
    python scripts/cli.py fetch
    ```
* **Custom arXiv Query:** Fetch specific arXiv papers and default web URLs.
    ```bash
    python scripts/cli.py fetch -aq "machine learning agents" -na 20
    ```
    (`-na`: number of arXiv results)
* **LLM Suggested Sources:** Let the LLM suggest sources based on a topic.
    ```bash
    python scripts/cli.py fetch --suggest-sources -t "Explainable AI Techniques"
    ```
    (`-t`: topic is required)

**2. Get Recommendations/Answers (`recommend`)**

* **Default Query (Strict RAG, Detailed):** Answer using default query from `config.py`, using only local data.
    ```bash
    python scripts/cli.py recommend
    ```
* **Custom Query (Strict RAG, Detailed):**
    ```bash
    python scripts/cli.py recommend -q "What are the challenges of training large language models?"
    ```
* **Hybrid RAG Mode (Allows LLM General Knowledge):**
    ```bash
    python scripts/cli.py recommend -q "Summarize recent advancements in AI agents." --general
    ```
* **Concise Output Format:** Get a structured summary instead of a detailed answer.
    ```bash
    python scripts/cli.py recommend -q "Key differences between supervised and unsupervised learning." --concise
    ```
* **Combine Modes (Hybrid + Concise):**
    ```bash
    python scripts/cli.py recommend -q "Explain Retrieval-Augmented Generation." --general --concise
    ```
* **Verbose Logging:** Add `-v` to any command for more detailed logs.
    ```bash
    python scripts/cli.py recommend -q "Test query" -v
    ```

**3. Find Papers on arXiv (`find_arxiv`)**

* Search arXiv directly without RAG or local processing.
    ```bash
    python scripts/cli.py find_arxiv -q "reinforcement learning from human feedback" -n 5
    ```
    (`-q`: query, `-n`: number of results)

## Key Dependencies
* Python 3.9+
* Playwright (for web crawling)
* google-generativeai (for Gemini Embeddings & LLM)
* readability-lxml (for HTML content extraction)
* PyMuPDF (for PDF text extraction)
* rank_bm25 (for keyword search)
* NLTK (for text processing)
* NumPy
* Requests
* BeautifulSoup4
