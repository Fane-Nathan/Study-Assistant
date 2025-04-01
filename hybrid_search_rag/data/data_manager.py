# hybrid_search_rag/data/data_manager.py
"""Handles saving and loading metadata, embeddings, and BM25 index."""

import os
import json
import numpy as np
import pickle
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class DataManager:
    """Manages loading and saving of resource metadata, embeddings, and BM25 index."""
    def __init__(self, data_dir: str, metadata_filename: str, embeddings_filename: str, bm25_filename: str):
        """Initializes the DataManager."""
        self.data_dir = data_dir
        self.metadata_path = os.path.join(data_dir, metadata_filename)
        self.embeddings_path = os.path.join(data_dir, embeddings_filename)
        self.bm25_path = os.path.join(data_dir, bm25_filename)
        try:
            # Ensure directory exists upon initialization
            os.makedirs(self.data_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create data directory {self.data_dir}: {e}")
            # Decide if this should be a fatal error depending on usage

    def save_all_data(self, metadata: List[Dict[str, Any]], embeddings: Optional[np.ndarray], bm25_index: Optional[object]):
        """Saves metadata, embeddings (if not None), and the BM25 index (if not None)."""
        if not metadata:
            logger.warning("Attempted to save empty metadata list. Aborting save.")
            return # Don't save if metadata is empty

        if embeddings is not None and len(metadata) != embeddings.shape[0]:
            logger.error(f"Metadata count ({len(metadata)}) and embeddings count ({embeddings.shape[0]}) mismatch during save.")
            raise ValueError("Metadata count and embeddings count must match.")

        try:
            # Save metadata
            logger.info(f"Saving metadata ({len(metadata)} items) to {self.metadata_path}")
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # Save embeddings
            if embeddings is not None:
                logger.info(f"Saving embeddings (shape: {embeddings.shape}) to {self.embeddings_path}")
                np.save(self.embeddings_path, embeddings, allow_pickle=False) # allow_pickle=False is safer
            else:
                logger.info(f"Embeddings data is None, skipping save to {self.embeddings_path}")
                # If embeddings file exists from previous run, remove it for consistency? Optional.
                if os.path.exists(self.embeddings_path):
                    try:
                        os.remove(self.embeddings_path)
                        logger.info(f"Removed existing embeddings file: {self.embeddings_path}")
                    except OSError as e:
                         logger.warning(f"Could not remove existing embeddings file {self.embeddings_path}: {e}")


            # Save BM25 index
            if bm25_index is not None:
                logger.info(f"Saving BM25 index to {self.bm25_path}")
                with open(self.bm25_path, 'wb') as f:
                    pickle.dump(bm25_index, f)
            else:
                logger.info(f"BM25 index is None, skipping save to {self.bm25_path}")
                # If index file exists from previous run, remove it? Optional.
                if os.path.exists(self.bm25_path):
                     try:
                         os.remove(self.bm25_path)
                         logger.info(f"Removed existing BM25 index file: {self.bm25_path}")
                     except OSError as e:
                          logger.warning(f"Could not remove existing BM25 index file {self.bm25_path}: {e}")

            logger.info("Data saving process completed successfully.")
        except IOError as e:
            logger.error(f"Failed to write data file: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during saving data: {e}", exc_info=True)
            raise

    def load_all_data(self) -> Tuple[Optional[List[Dict[str, Any]]], Optional[np.ndarray], Optional[object]]:
        """Loads metadata, embeddings, and BM25 index. Returns None for missing/failed parts."""
        metadata: Optional[List[Dict[str, Any]]] = None
        embeddings: Optional[np.ndarray] = None
        bm25_index: Optional[object] = None

        # --- Load Metadata (Essential) ---
        if not os.path.exists(self.metadata_path):
            logger.warning(f"Metadata file not found: {self.metadata_path}.")
            return metadata, embeddings, bm25_index
        try:
            logger.info(f"Loading metadata from {self.metadata_path}")
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata_content = json.load(f)
            if not isinstance(metadata_content, list):
                 logger.error("Metadata file does not contain a valid JSON list.")
                 return None, embeddings, bm25_index # Metadata failed critically
            metadata = metadata_content # Assign only if valid
            logger.info(f"Loaded {len(metadata)} metadata items.")
        except (json.JSONDecodeError, IOError) as e:
             logger.error(f"Failed to load or decode metadata: {e}")
             return None, embeddings, bm25_index # Critical failure
        except Exception as e:
             logger.error(f"Unexpected error loading metadata: {e}", exc_info=True)
             return None, embeddings, bm25_index

        # --- Load Embeddings (Optional) ---
        if not os.path.exists(self.embeddings_path):
            logger.warning(f"Embeddings file not found: {self.embeddings_path}.")
        else:
            try:
                logger.info(f"Loading embeddings from {self.embeddings_path}")
                embeddings = np.load(self.embeddings_path, allow_pickle=False) # allow_pickle=False is safer
                # Consistency Check
                if metadata and len(metadata) != embeddings.shape[0]:
                    logger.error(f"Metadata count ({len(metadata)}) and embeddings count ({embeddings.shape[0]}) differ. Discarding loaded embeddings.")
                    embeddings = None # Invalidate due to mismatch
                elif metadata: # Only log shape if metadata exists for comparison
                    logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
                else: # Embeddings loaded but no metadata? Problem.
                     logger.error("Loaded embeddings but metadata is missing. Discarding embeddings.")
                     embeddings = None

            except (IOError, ValueError) as e:
                logger.error(f"Failed to load embeddings file: {e}")
                embeddings = None
            except Exception as e:
                 logger.error(f"Unexpected error loading embeddings: {e}", exc_info=True)
                 embeddings = None

        # --- Load BM25 Index (Optional) ---
        if not os.path.exists(self.bm25_path):
             logger.warning(f"BM25 index file not found: {self.bm25_path}.")
        else:
            try:
                logger.info(f"Loading BM25 index from {self.bm25_path}")
                with open(self.bm25_path, 'rb') as f:
                    bm25_index = pickle.load(f)
                logger.info("Loaded BM25 index.")
                # Add check: Is the loaded object actually a BM25 object? Requires rank_bm25 import
                # from rank_bm25 import BM25Okapi # Requires adding to top imports
                # if not isinstance(bm25_index, BM25Okapi):
                #      logger.error("Loaded BM25 file does not contain a valid BM25Okapi object. Discarding index.")
                #      bm25_index = None
            except (IOError, pickle.UnpicklingError, AttributeError, EOFError) as e: # More specific pickle errors
                logger.error(f"Failed to load or unpickle BM25 index: {e}")
                bm25_index = None
            except Exception as e:
                logger.error(f"Unexpected error loading BM25 index: {e}", exc_info=True)
                bm25_index = None

        logger.info("Data loading process completed.")
        return metadata, embeddings, bm25_index