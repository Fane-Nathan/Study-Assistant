# HybridSearchRAG

A project demonstrating hybrid search (semantic + keyword) and Retrieval-Augmented Generation (RAG) using local documents (arXiv papers, web articles) and external LLM APIs.

## Features

* Fetches data from arXiv and specified web URLs.
* Generates embeddings using Sentence Transformers.
* Builds a BM25 index for keyword search.
* Performs hybrid retrieval using Reciprocal Rank Fusion (RRF).
* Supports two RAG modes via Hugging Face Inference API:
    * Strict RAG (answers based only on local documents).
    * Hybrid-Knowledge RAG (answers use local documents + LLM general knowledge via `--general` flag).
* Command-line interface (`scripts/cli.py`).

## Setup

1.  **Clone/Create Project:** Get the code into your `HybridSearchRAG` directory.
2.  **Create `.env` file:** Copy the content from the provided `.env` section and paste your actual Hugging Face API token.
3.  **Create & Activate Environment:** (Using Conda example)
    ```bash
    conda create --name hybrid-rag python=3.9 -y
    conda activate hybrid-rag
    ```
    *(Adjust python version if needed)*
4.  **Install Dependencies:** Navigate to the `HybridSearchRAG` root directory and run:
    ```bash
    pip install -r requirements.txt
    ```
5.  **Download NLTK Data:**
    Run Python interpreter (`python`) and execute:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    # nltk.download('punkt_tab') # Usually not needed unless specific tokenizer issues arise
    ```
6.  **Configure Data Sources:** Edit `hybrid_search_rag/config.py`. **Crucially, update `TARGET_WEB_URLS`** with valid URLs you want to scrape. Adjust other settings like `DEFAULT_ARXIV_QUERY`, model IDs, or RAG parameters if desired.
7.  **Fetch Initial Data:** Run the fetch command from the project root to populate the `data/` directory.
    ```bash
    python scripts/cli.py fetch
    ```

## Usage

Run commands from the `HybridSearchRAG` root directory.

* **Strict RAG (Default):** Answers based *only* on retrieved local documents.
    ```bash
    python scripts/cli.py recommend -q "Your question about fetched documents"
    ```
* **Hybrid-Knowledge RAG:** Answers using local documents as context *plus* the LLM's general knowledge.
    ```bash
    python scripts/cli.py recommend --general -q "Your broader question"
    ```
* **Force Data Fetch:** Re-downloads and processes data from sources defined in `config.py`.
    ```bash
    python scripts/cli.py fetch
    ```
* **Force Fetch with Options:** Override default arXiv query and number of results.
    ```bash
    python scripts/cli.py fetch -aq "specific topic" -na 30
    ```
* **Get Help:**
    ```bash
    python scripts/cli.py --help
    python scripts/cli.py recommend --help
    python scripts/cli.py fetch --help
    ```