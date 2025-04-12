# hybrid_search_rag/retrieval/recommender.py
"""Handles hybrid recommendation using semantic search (embeddings) and keyword search (BM25),
fusing the results using Reciprocal Rank Fusion (RRF)."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity # For semantic search
from rank_bm25 import BM25Okapi # For keyword search
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import logging
import nltk # For tokenization and stopwords
from nltk.corpus import stopwords
import string # For punctuation removal

# Import the configured EmbeddingModel implementation
from .embedding_model_gemini import EmbeddingModel # Assumes Gemini is the configured model


@dataclass
class RecommendationParams:
    """Parameters for recommendation."""
    semantic_candidates: int
    keyword_candidates: int
    fusion_k: int
    top_n_final: int
logger = logging.getLogger(__name__)

class NltkManager:
    """Manages NLTK data and tokenization."""
    NLTK_STOPWORDS: Optional[set] = None
    NLTK_DATA_AVAILABLE: Dict[str, bool] = {'punkt': False, 'stopwords': False}

    @classmethod
    def _check_and_load_nltk_data(cls):
        """Checks and potentially downloads NLTK data required for tokenization."""
        data_to_check = {'punkt': 'tokenizers/punkt', 'stopwords': 'corpora/stopwords'}
        needs_download = []

        for name, path in data_to_check.items():
            try:
                nltk.data.find(path)
                cls.NLTK_DATA_AVAILABLE[name] = True
            except LookupError:
                cls.NLTK_DATA_AVAILABLE[name] = False
                needs_download.append(name)
                logger.warning(f"NLTK data '{name}' not found. Attempting download.")

        if needs_download:
            logger.info(f"Downloading missing NLTK data: {', '.join(needs_download)}...")
            try:
                for name in needs_download:
                    if nltk.download(name, quiet=True):
                        logger.info(f"Successfully downloaded NLTK data '{name}'.")
                        cls.NLTK_DATA_AVAILABLE[name] = True
                    else:
                        logger.error(f"NLTK download command failed for '{name}'.", exc_info=True)
            except Exception as e:
                logger.error(f"NLTK download failed: {e}", exc_info=True)

        # Load stopwords if available
        if cls.NLTK_DATA_AVAILABLE['stopwords']:
            try:
                cls.NLTK_STOPWORDS = set(stopwords.words('english'))
                logger.info(f"Loaded {len(cls.NLTK_STOPWORDS)} NLTK English stopwords.")
            except Exception as e:
                logger.error(f"Failed to load NLTK stopwords: {e}", exc_info=True)
                cls.NLTK_STOPWORDS = set()  # Fallback to empty set
        else:
            logger.warning("NLTK stopwords data unavailable. Using empty stopword list.")
            cls.NLTK_STOPWORDS = set()

    @classmethod
    def tokenize_text(cls, text: str, remove_stopwords: bool = True, min_word_length: int = 2) -> List[str]:
        """Tokenizes text, optionally removes stopwords and short words.
        Args:
            text: The text to tokenize.
            remove_stopwords: Whether to remove stopwords (default: True).
            min_word_length: Minimum word length to keep (default: 2).
        """
        if not isinstance(text, str):
            logger.warning("Attempted to tokenize non-string input.")
            return []
        try:
            if not cls.NLTK_DATA_AVAILABLE['punkt']:
                logger.error("NLTK 'punkt' data unavailable! Cannot tokenize for keyword search.", exc_info=True)                
                return []
            text = text.lower()
            # Remove punctuation using translation table
            text = text.translate(str.maketrans('', '', string.punctuation))
            tokens = nltk.word_tokenize(text)
            if remove_stopwords:
                tokens = [word for word in tokens if word not in (cls.NLTK_STOPWORDS or set())]
            return [word for word in tokens if len(word) >= min_word_length]
        except Exception as e:
            logger.error(f"Tokenization failed for text snippet: '{text[:50]}...': {e}", exc_info=True)
            return []

class Recommender:
    """Orchestrates hybrid search using embeddings and BM25, fused with RRF."""

    def __init__(self, embed_model: EmbeddingModel):
        """Initializes with a pre-configured EmbeddingModel instance."""
        self.embed_model = embed_model
        NltkManager._check_and_load_nltk_data()
        self.nltk_manager = NltkManager()
        logger.info(f"Recommender initialized with embedding model: {type(embed_model).__name__}")

    def _validate_embeddings(self, query_embedding: Optional[np.ndarray], resource_embeddings: Optional[np.ndarray]) -> bool:
        """Helper function to validate embeddings for semantic search."""
        if resource_embeddings is None or query_embedding is None or query_embedding.size == 0 or resource_embeddings.size == 0:
            return False
        
        # Ensure query embedding is 2D (1, embedding_dim)
        query_embedding_2d = query_embedding.reshape(1, -1) if query_embedding.ndim == 1 else query_embedding
        if query_embedding_2d.ndim != 2 or query_embedding_2d.shape[0] != 1:
            logger.error(f"Invalid query embedding shape: {query_embedding.shape}. Aborting semantic search.", exc_info=True)
            return False
        if resource_embeddings.ndim != 2 or resource_embeddings.shape[1] != query_embedding_2d.shape[1]:
            logger.error(f"Embeddings shape mismatch: Resource={resource_embeddings.shape}, Query={query_embedding_2d.shape}. Aborting.", exc_info=True)
            return False
        return True
    
    def _validate_keyword_search_inputs(self, query: str, bm25_index: Optional[BM25Okapi]) -> Tuple[bool, List[str]]:
        """Helper function to validate inputs for keyword search."""
        if bm25_index is None:
            return False, []
        
        tokenized_query = self.nltk_manager.tokenize_text(query, remove_stopwords=True, min_word_length=2)
        if not tokenized_query:
            logger.warning(f"Query '{query[:50]}...' became empty after tokenization. Skipping keyword search.")
            return False, []
        return True, tokenized_query

    def _semantic_search(self, query_embedding: Optional[np.ndarray], resource_embeddings: Optional[np.ndarray], top_n: int) -> List[Tuple[int, float]]:
        """Performs semantic search using cosine similarity."""
        if not self._validate_embeddings(query_embedding, resource_embeddings):
            return []
        try:

            query_embedding_2d = query_embedding.reshape(1, -1) if query_embedding.ndim == 1 else query_embedding
            similarities = cosine_similarity(query_embedding_2d, resource_embeddings)[0]
            # Find top N indices efficiently using argpartition
            num_results = min(top_n, len(similarities))
            if num_results <= 0: return []
            # Get indices of top N elements (unsorted among themselves)
            top_indices_unsorted = np.argpartition(similarities, -num_results)[-num_results:]
            # Sort only the top N indices by score (descending)
            top_indices_sorted = top_indices_unsorted[np.argsort(similarities[top_indices_unsorted])[::-1]]

            # Format results: (index, score) tuples
            results = [(int(i), float(similarities[i])) for i in top_indices_sorted]
            # logger.debug(f"Semantic search returned {len(results)} candidates.") # Debug level
            return results
        except Exception as e:
             logger.error(f"Error during semantic search: {e}", exc_info=True)
             return []

    def _keyword_search(self, query: str, bm25_index: Optional[BM25Okapi], num_docs_in_corpus: int, top_n: int) -> List[Tuple[int, float]]:
        """Performs keyword search using the BM25 index."""
        valid_inputs, tokenized_query = self._validate_keyword_search_inputs(query, bm25_index)
        if not valid_inputs: return []
        try:
            

            # Get BM25 scores for the query against all documents
            doc_scores = bm25_index.get_scores(tokenized_query)

            # Find top N indices efficiently using argpartition
            num_results = min(top_n, len(doc_scores))
            if num_results <= 0: return []
            top_indices_unsorted = np.argpartition(doc_scores, -num_results)[-num_results:]
            # Sort only the top N indices by score (descending)
            top_indices_sorted = top_indices_unsorted[np.argsort(doc_scores[top_indices_unsorted])[::-1]]

            # Format results: (index, score) tuples
            results = [(int(i), float(doc_scores[i])) for i in top_indices_sorted]
            # logger.debug(f"Keyword search returned {len(results)} candidates.") # Debug level
            return results
        except Exception as e:
             logger.error(f"Error during keyword search: {e}", exc_info=True)
             return []

    def _reciprocal_rank_fusion(self, ranked_lists: List[List[Tuple[int, float]]], k: int = 60) -> Dict[int, float]:
        """Combines multiple ranked lists using Reciprocal Rank Fusion (RRF)."""
        fused_scores: Dict[int, float] = {}
        if not ranked_lists: return fused_scores

        # logger.debug(f"Performing RRF (k={k}) on {len(ranked_lists)} lists...") # Debug level
        for rank_list in ranked_lists:
            if not rank_list: continue
            seen_indices_in_list = set()
            for rank, (doc_index, _) in enumerate(rank_list): # rank starts at 0
                 if isinstance(doc_index, (int, np.integer)) and doc_index >= 0: # add test to check doc_index >0
                      doc_index_int = int(doc_index)
                      if doc_index_int not in seen_indices_in_list:
                           # RRF Formula: score += 1 / (k + rank + 1)
                           fused_scores[doc_index_int] = fused_scores.get(doc_index_int, 0.0) + (1.0 / (k + rank + 1))
                           seen_indices_in_list.add(doc_index_int)
                 else:
                      logger.warning(f"Skipping invalid doc_index ({doc_index}) during RRF.", exc_info=True)
        # logger.debug(f"Fusion generated {len(fused_scores)} scores.") # Debug level
        return fused_scores

    def recommend(self,
                  query: str,
                  resource_metadata: List[Dict[str, Any]],
                  resource_embeddings: Optional[np.ndarray],
                  bm25_index: Optional[BM25Okapi],
                  params: RecommendationParams) -> List[Tuple[Dict[str, Any], float]]:
        """Orchestrates hybrid search: encode query, run searches, fuse, return top N."""
        if not resource_metadata:
             logger.warning("Cannot recommend: Resource metadata is empty.")
             return []

        logger.info(f"Starting hybrid recommendation for query: '{query[:100]}...'")
        num_docs = len(resource_metadata)

        # 1. Encode Query (using appropriate task type)
        use_semantic_search = True
        query_embedding: Optional[np.ndarray] = None
        # logger.debug(f"Encoding query (task_type='RETRIEVAL_QUERY').") # Debug level
        try:
            # Use 'RETRIEVAL_QUERY' task type for optimal query embeddings
            query_embedding = self.embed_model.encode(query, task_type="RETRIEVAL_QUERY")
            if query_embedding is None or query_embedding.size == 0:
                logger.warning("Query encoding failed or resulted in empty vector. Semantic search disabled.")
                use_semantic_search = False
        except Exception as e:
            logger.error(f"Error during query encoding: {e}", exc_info=True)
            use_semantic_search = False

        # 2. Semantic Search
        # logger.info(f"Performing semantic search (Top {semantic_candidates})...") # A bit verbose
        semantic_results = self._semantic_search(query_embedding, resource_embeddings, params.semantic_candidates)
        logger.info(f"Semantic search returned {len(semantic_results)} candidates.")

        # 3. Keyword Search
        # logger.info(f"Performing keyword search (Top {keyword_candidates})...") # A bit verbose
        keyword_results = self._keyword_search(query, bm25_index, num_docs, params.keyword_candidates)
        logger.info(f"Keyword search returned {len(keyword_results)} candidates.")

        # 4. Fuse Results
        logger.info(f"Fusing search results (k={params.fusion_k})...")
        lists_to_fuse = [lst for lst in [semantic_results, keyword_results] if lst]
        if not lists_to_fuse:
             logger.warning("No candidates found from any search method. Returning empty results.")
             return []

        fused_scores = self._reciprocal_rank_fusion(lists_to_fuse, k=params.fusion_k)
        if not fused_scores:
             logger.warning("Fusion resulted in empty scores dictionary. Returning empty results.")
             return []        

        # 5. Get Top N Fused Results (Sort by score)
        # logger.debug(f"Sorting {len(fused_scores)} fused results...") # Debug level
        sorted_fused_indices = sorted(fused_scores.keys(), key=lambda idx: fused_scores[idx], reverse=True)

        # 6. Format Final Output (Metadata + Score)
        # logger.info(f"Retrieving metadata for top {top_n_final} results...") # Can be verbose
        final_results: List[Tuple[Dict[str, Any], float]] = []
        for rank, idx in enumerate(sorted_fused_indices):
            if len(final_results) >= params.top_n_final: break # Stop once we have enough
            if 0 <= idx < num_docs: # Check index validity
                try:
                    metadata_item = resource_metadata[idx]
                    score = fused_scores[idx]
                    final_results.append((metadata_item, float(score)))
                except IndexError:
                     logger.error(f"IndexError accessing metadata at index {idx}. Skipping.", exc_info=True)
                except Exception as e:
                     logger.error(f"Error formatting result for index {idx}: {e}", exc_info=True)
            else:
                logger.error(f"Invalid index {idx} encountered after fusion (range [0, {num_docs-1}]). Skipping.", exc_info=True)

        logger.info(f"Recommendation complete. Returning {len(final_results)} final results.")
        return final_results