# hybrid_search_rag/retrieval/recommender.py
"""Handles hybrid recommendation logic using semantic search and BM25."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple, Optional
import logging
import nltk
from nltk.corpus import stopwords
import string
import importlib.resources # For checking NLTK data

# Import EmbeddingModel relative to the package structure
# --- Ensure this points to the correct embedding model implementation ---
from .embedding_model_gemini import EmbeddingModel
# ---

logger = logging.getLogger(__name__)

# --- NLTK Data Handling ---
# (NLTK code remains unchanged - assuming it's correct)
NLTK_STOPWORDS = None
NLTK_DATA_AVAILABLE = {'punkt': False, 'stopwords': False}

def _check_and_load_nltk_data():
    """Checks required NLTK data and attempts download if missing."""
    global NLTK_STOPWORDS # Allow modification of global variable

    data_to_check = {'punkt': 'tokenizers/punkt', 'stopwords': 'corpora/stopwords'}
    needs_download = []

    for name, path in data_to_check.items():
        try:
            nltk.data.find(path)
            NLTK_DATA_AVAILABLE[name] = True
            logger.debug(f"NLTK data '{name}' found.")
        except LookupError:
            NLTK_DATA_AVAILABLE[name] = False
            needs_download.append(name)
            logger.warning(f"NLTK data '{name}' not found.")

    if needs_download:
        logger.info(f"Attempting to download missing NLTK data: {', '.join(needs_download)}...")
        try:
            for name in needs_download:
                if nltk.download(name, quiet=True):
                    logger.info(f"Successfully downloaded NLTK data '{name}'.")
                    NLTK_DATA_AVAILABLE[name] = True
                else:
                    logger.error(f"Failed to download NLTK data '{name}'.")
        except Exception as e:
            logger.error(f"An error occurred during NLTK download: {e}")

    # Load stopwords
    if NLTK_DATA_AVAILABLE['stopwords']:
        try:
            NLTK_STOPWORDS = set(stopwords.words('english'))
            logger.info("NLTK stopwords loaded.")
        except Exception as e:
             logger.error(f"Failed to load NLTK stopwords even after potential download: {e}")
             NLTK_STOPWORDS = set() # Fallback
    else:
        NLTK_STOPWORDS = set() # Fallback if download failed

# --- Tokenizer for BM25 ---
# (tokenize_text function remains unchanged - assuming it's correct)
def tokenize_text(text: str) -> List[str]:
    """Basic tokenizer: lowercases, removes punctuation, removes stopwords."""
    if not isinstance(text, str): return []
    try:
        if not NLTK_DATA_AVAILABLE['punkt']:
             logger.error("Cannot tokenize: NLTK 'punkt' data is unavailable.")
             return [] # Cannot proceed without tokenizer data
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = nltk.word_tokenize(text)
        # Use the loaded stopwords set (might be empty if download failed)
        current_stopwords = NLTK_STOPWORDS if NLTK_STOPWORDS is not None else set()
        return [word for word in tokens if word not in current_stopwords and len(word) > 1]
    except Exception as e:
        logger.error(f"Error during tokenization of text snippet starting with '{text[:50]}...': {e}", exc_info=True)
        return []


class Recommender:
    """Calculates hybrid similarity using embeddings and BM25, then fuses ranks."""

    def __init__(self, embed_model: EmbeddingModel):
        """
        Initializes the Recommender.
        Relies on the provided EmbeddingModel instance being configured correctly.
        """
        self.embed_model = embed_model
        # --- MODIFICATION START ---
        # Removed the check: if self.embed_model.model is None:
        # The new embedding_model_gemini handles its configuration internally
        # during its own __init__. If it fails there, it should raise an error.
        logger.info("Recommender initialized with EmbeddingModel.")
        # --- MODIFICATION END ---

    def _semantic_search(self, query_embedding: Optional[np.ndarray], resource_embeddings: Optional[np.ndarray], top_n: int) -> List[Tuple[int, float]]:
        """Performs semantic search and returns (index, score) tuples."""
        # (This method remains unchanged)
        if resource_embeddings is None or query_embedding is None or resource_embeddings.size == 0:
            logger.debug("Skipping semantic search due to missing embeddings.")
            return []
        try:
            # Ensure query_embedding is 2D for cosine_similarity
            if query_embedding.ndim == 1:
                 query_embedding_2d = query_embedding.reshape(1, -1)
            elif query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
                 query_embedding_2d = query_embedding
            else:
                 logger.error(f"Unexpected query embedding shape: {query_embedding.shape}")
                 return []

            similarities = cosine_similarity(query_embedding_2d, resource_embeddings)[0]
            num_results = min(top_n, len(similarities))
            if num_results <= 0: return []

            # Get indices of top N scores
            # Using partition is slightly more efficient than full sort for finding top N
            # Ensure we handle cases where num_results is larger than available similarities
            actual_top_n = min(num_results, len(similarities))
            if actual_top_n > 0:
                 top_indices = np.argpartition(similarities, -actual_top_n)[-actual_top_n:]
                 # Sort only the top N indices by score
                 top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
            else:
                 top_indices = np.array([], dtype=int)


            # Create results list (index, score)
            # Added a small threshold to filter out potential zero/negative similarities if needed
            results = [(int(i), float(similarities[i])) for i in top_indices if similarities[i] > 0.0]
            logger.debug(f"Semantic search returned {len(results)} results.")
            return results
        except Exception as e:
             logger.error(f"Error during semantic search: {e}", exc_info=True)
             return []

    def _keyword_search(self, query: str, bm25_index: Optional[BM25Okapi], num_docs_in_corpus: int, top_n: int) -> List[Tuple[int, float]]:
        """Performs keyword search using BM25 and returns (index, score) tuples."""
        # (This method remains unchanged)
        if bm25_index is None:
            logger.debug("Skipping keyword search due to missing BM25 index.")
            return []
        try:
            tokenized_query = tokenize_text(query)
            if not tokenized_query:
                 logger.warning("Query tokenization resulted in empty query for keyword search.")
                 return []

            # Check for potential mismatch if num_docs_in_corpus provided differs
            # It's better practice if the caller ensures consistency, but warning is good.
            if hasattr(bm25_index, 'corpus_size') and num_docs_in_corpus != bm25_index.corpus_size:
                 logger.warning(f"Num docs provided ({num_docs_in_corpus}) differs from BM25 corpus size ({bm25_index.corpus_size}). Using BM25 size for safety.")
                 # It might be safer to rely on the index's reported size if available
                 # num_docs_in_corpus = bm25_index.corpus_size # Or raise error

            # BM25 scores can be negative for non-matching docs, get_scores gives all
            doc_scores = bm25_index.get_scores(tokenized_query)

            # Get top N overall scores
            num_results = min(top_n, len(doc_scores))
            if num_results <= 0: return []

            # Using partition like in semantic search
            actual_top_n = min(num_results, len(doc_scores))
            if actual_top_n > 0:
                 top_indices = np.argpartition(doc_scores, -actual_top_n)[-actual_top_n:]
                 top_indices = top_indices[np.argsort(doc_scores[top_indices])[::-1]]
            else:
                 top_indices = np.array([], dtype=int)


            # Create results list (index, score) - Optionally filter out scores <= 0
            results = [(int(i), float(doc_scores[i])) for i in top_indices] # if doc_scores[i] > 0.0]
            logger.debug(f"Keyword search returned {len(results)} results.")
            return results

        except Exception as e:
             logger.error(f"Error during keyword search: {e}", exc_info=True)
             return []

    def _reciprocal_rank_fusion(self, ranked_lists: List[List[Tuple[int, float]]], k: int = 60) -> Dict[int, float]:
        """Performs Reciprocal Rank Fusion (RRF) on multiple ranked lists."""
        # (This method remains unchanged)
        fused_scores: Dict[int, float] = {}
        if not ranked_lists: return fused_scores

        for rank_list in ranked_lists:
            if not rank_list: continue
            seen_indices_in_list = set() # Avoid double counting from same list if duplicates somehow occur
            for rank, (doc_index, _) in enumerate(rank_list):
                 # Ensure doc_index is a valid integer before using
                 if isinstance(doc_index, (int, np.integer)) and doc_index >= 0:
                      doc_index_int = int(doc_index) # Convert numpy int if needed
                      if doc_index_int not in seen_indices_in_list:
                           # RRF formula: score += 1 / (k + rank) where rank starts at 1
                           # Since enumerate gives rank starting at 0, rank+1 is the correct rank value
                           fused_scores[doc_index_int] = fused_scores.get(doc_index_int, 0.0) + (1.0 / (k + rank + 1))
                           seen_indices_in_list.add(doc_index_int)
                      # else: # Optionally log duplicates within the same list
                      #      logger.debug(f"Duplicate index {doc_index_int} found within a single ranked list during RRF.")
                 else:
                      logger.warning(f"Skipping invalid or negative doc_index during RRF: {doc_index}")
        return fused_scores

    def recommend(self,
                  query: str,
                  resource_metadata: List[Dict[str, Any]],
                  resource_embeddings: Optional[np.ndarray],
                  bm25_index: Optional[BM25Okapi],
                  semantic_candidates: int,
                  keyword_candidates: int,
                  fusion_k: int,
                  top_n_final: int) -> List[Tuple[Dict[str, Any], float]]:
        """Performs hybrid search and fuses results."""

        if not resource_metadata:
             logger.warning("No resource metadata provided for recommendation.")
             return []

        logger.info(f"Generating hybrid recommendations for query: '{query}'")
        num_docs = len(resource_metadata)

        # 1. Encode Query using appropriate task type
        query_embedding: Optional[np.ndarray] = None
        try:
            # --- MODIFICATION START ---
            logger.debug(f"Encoding query for semantic search (task_type='RETRIEVAL_QUERY').")
            # Explicitly pass the task_type for query encoding
            query_embedding = self.embed_model.encode(query, task_type="RETRIEVAL_QUERY")
            # --- MODIFICATION END ---

            if query_embedding is None:
                 # Log error if encode returned None explicitly
                 logger.error("Query embedding failed (returned None). Skipping semantic search.")
            elif query_embedding.size == 0:
                 # Handle case where encode might return an empty array
                 logger.warning("Query embedding resulted in an empty array. Skipping semantic search.")
                 query_embedding = None # Ensure it's None for semantic search check

        except Exception as e:
             # Catch potential errors during the encode call itself
             logger.error(f"An unexpected error occurred during query encoding: {e}", exc_info=True)
             query_embedding = None # Ensure semantic search is skipped on error

        # 2. Perform Semantic Search
        # This will naturally be skipped if query_embedding is None
        semantic_results = self._semantic_search(query_embedding, resource_embeddings, semantic_candidates)

        # 3. Perform Keyword Search
        keyword_results = self._keyword_search(query, bm25_index, num_docs, keyword_candidates)

        # 4. Fuse Results
        logger.info(f"Fusing results (Semantic candidates found: {len(semantic_results)}, Keyword candidates found: {len(keyword_results)})...")
        lists_to_fuse = [lst for lst in [semantic_results, keyword_results] if lst] # Filter out empty lists
        if not lists_to_fuse:
             logger.warning("No candidates found from any search method after filtering.")
             return [] # Return empty if both searches yielded nothing

        fused_scores = self._reciprocal_rank_fusion(lists_to_fuse, k=fusion_k)

        # 5. Get Top N Fused Results
        if not fused_scores:
             logger.warning("Fusion resulted in empty scores, though candidates existed.") # Should not happen if lists_to_fuse was not empty
             return []
        # Sort the document indices based on their fused scores in descending order
        sorted_fused_indices = sorted(fused_scores.keys(), key=lambda idx: fused_scores[idx], reverse=True)

        # 6. Format Final Results with Metadata
        final_results: List[Tuple[Dict[str, Any], float]] = []
        for rank, idx in enumerate(sorted_fused_indices):
            # Stop adding results if we've reached the desired final count
            if len(final_results) >= top_n_final:
                break
            # Validate index against metadata length
            if 0 <= idx < num_docs:
                try:
                    metadata_item = resource_metadata[idx]
                    score = fused_scores[idx]
                    # Ensure score is float, just in case
                    final_results.append((metadata_item, float(score)))
                except IndexError:
                     logger.error(f"IndexError accessing metadata at index {idx} even though it was within range [0, {num_docs-1}]. Skipping.")
                except Exception as e:
                     logger.error(f"Unexpected error formatting result for index {idx}: {e}", exc_info=True)

            else:
                # This case indicates an issue upstream (e.g., in fusion or indexing)
                logger.warning(f"Fused index {idx} out of range [0, {num_docs-1}] for metadata. Skipping.")

        logger.info(f"Returning top {len(final_results)} fused recommendations.")
        return final_results