# hybrid_search_rag/data/data_manager.py
"""Handles saving and loading project data: metadata, embeddings, and BM25 index."""

import os
import json
import numpy as np
import pickle # For object serialization (BM25 index)
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class DataManager:
    """Manages loading and saving of resource metadata, embeddings, and BM25 index."""
    def __init__(self, data_dir: str, metadata_filename: str, embeddings_filename: str, bm25_filename: str):
        """Initializes paths and ensures the data directory exists."""
        self.data_dir = data_dir
        self.metadata_path = os.path.join(data_dir, metadata_filename)
        self.embeddings_path = os.path.join(data_dir, embeddings_filename)
        self.bm25_path = os.path.join(data_dir, bm25_filename)
        try:
            # Ensure data directory exists, creating it if necessary.
            os.makedirs(self.data_dir, exist_ok=True)
        except OSError as e:
            # Failure to create directory is often critical.
            logger.error(f"Could not create data directory {self.data_dir}: {e}")
            # Consider raising an error depending on application requirements.

    def save_all_data(self, metadata: List[Dict[str, Any]], embeddings: Optional[np.ndarray], bm25_index: Optional[object]):
        """Saves metadata (JSON), embeddings (NumPy), and BM25 index (pickle)."""
        if not metadata:
            logger.warning("Attempted to save empty metadata list. Skipping save.")
            return

        # Sanity check: Ensure consistency between metadata and embeddings counts.
        if embeddings is not None and len(metadata) != embeddings.shape[0]:
            logger.error(f"Metadata count ({len(metadata)}) mismatch with embeddings count ({embeddings.shape[0]})! Aborting save.")
            raise ValueError("Metadata and embeddings counts must match.")

        try:
            # Save metadata
            logger.info(f"Saving metadata ({len(metadata)} items) to {self.metadata_path}")
            # 'w' mode for write, 'utf-8' for broad compatibility, indent for readability.
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # Save embeddings
            if embeddings is not None:
                logger.info(f"Saving embeddings (shape: {embeddings.shape}) to {self.embeddings_path}")
                # Use np.save; allow_pickle=False is recommended for security.
                np.save(self.embeddings_path, embeddings, allow_pickle=False)
            else:
                logger.info(f"No embeddings data provided, skipping save to {self.embeddings_path}")
                # Clean up old file if it exists but isn't being replaced this time.
                if os.path.exists(self.embeddings_path):
                    try:
                        os.remove(self.embeddings_path)
                        logger.info(f"Removed existing embeddings file: {self.embeddings_path}")
                    except OSError as e:
                         logger.warning(f"Could not remove existing embeddings file {self.embeddings_path}: {e}")

            # Save BM25 index
            if bm25_index is not None:
                logger.info(f"Saving BM25 index to {self.bm25_path}")
                # 'wb' (Write Binary) mode is required for pickle.
                with open(self.bm25_path, 'wb') as f:
                    pickle.dump(bm25_index, f)
            else:
                logger.info(f"No BM25 index provided, skipping save to {self.bm25_path}")
                # Clean up old file if it exists.
                if os.path.exists(self.bm25_path):
                     try:
                         os.remove(self.bm25_path)
                         logger.info(f"Removed existing BM25 index file: {self.bm25_path}")
                     except OSError as e:
                          logger.warning(f"Could not remove existing BM25 index file {self.bm25_path}: {e}")

            logger.info("Data saving process completed.")
        except IOError as e:
            # Catches file system errors during write operations.
            logger.error(f"IOError during file writing: {e}")
            raise
        except Exception as e:
            # Catch-all for other potential errors during saving.
            logger.error(f"Unexpected error during data saving: {e}", exc_info=True)
            raise

    def load_all_data(self) -> Tuple[Optional[List[Dict[str, Any]]], Optional[np.ndarray], Optional[object]]:
        """Loads metadata, embeddings, and BM25 index from disk."""
        metadata: Optional[List[Dict[str, Any]]] = None
        embeddings: Optional[np.ndarray] = None
        bm25_index: Optional[object] = None

        # --- Load Metadata (Essential) ---
        if not os.path.exists(self.metadata_path):
            logger.warning(f"Metadata file missing: {self.metadata_path}. Cannot load data.")
            return None, None, None
        try:
            logger.info(f"Loading metadata from {self.metadata_path}")
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata_content = json.load(f)
            # Basic validation: Ensure it's a list as expected.
            if not isinstance(metadata_content, list):
                 logger.error("Metadata file format error: Expected a JSON list.")
                 return None, None, None
            metadata = metadata_content
            logger.info(f"Loaded {len(metadata)} metadata items.")
        except (json.JSONDecodeError, IOError) as e:
             logger.error(f"Failed to load or decode metadata file: {e}")
             return None, None, None
        except Exception as e:
             logger.error(f"Unexpected error loading metadata: {e}", exc_info=True)
             return None, None, None

        # --- Load Embeddings (Optional Component) ---
        if not os.path.exists(self.embeddings_path):
            logger.warning(f"Embeddings file not found: {self.embeddings_path}. Semantic search disabled.")
        else:
            try:
                logger.info(f"Loading embeddings from {self.embeddings_path}")
                embeddings = np.load(self.embeddings_path, allow_pickle=False)
                # Sanity check: Compare loaded embeddings count with metadata count.
                if metadata and len(metadata) != embeddings.shape[0]:
                    logger.error(f"Data mismatch: Metadata count ({len(metadata)}) differs from loaded embeddings count ({embeddings.shape[0]}). Discarding embeddings.")
                    embeddings = None # Invalidate due to mismatch.
                elif metadata:
                    logger.info(f"Loaded embeddings with shape: {embeddings.shape}.")
                else: # Should not happen if metadata loading is mandatory
                     logger.error("Loaded embeddings but metadata is invalid?! Discarding embeddings.")
                     embeddings = None
            except (IOError, ValueError) as e: # Handles file read errors or corrupt .npy files
                logger.error(f"Failed to load embeddings file (corrupt?): {e}")
                embeddings = None
            except Exception as e:
                 logger.error(f"Unexpected error loading embeddings: {e}", exc_info=True)
                 embeddings = None

        # --- Load BM25 Index (Optional Component) ---
        if not os.path.exists(self.bm25_path):
             logger.warning(f"BM25 index file not found: {self.bm25_path}. Keyword search disabled.")
        else:
            try:
                logger.info(f"Loading BM25 index from {self.bm25_path}")
                # 'rb' for Read Binary mode.
                with open(self.bm25_path, 'rb') as f:
                    bm25_index = pickle.load(f)
                logger.info("Loaded BM25 index.")
                # Optional: Add type checking here if desired, e.g., isinstance(bm25_index, BM25Okapi)
            except (IOError, pickle.UnpicklingError, AttributeError, EOFError, ImportError, IndexError) as e:
                # Catch various errors related to file reading or unpickling issues.
                logger.error(f"Failed to load or unpickle BM25 index: {e}")
                bm25_index = None
            except Exception as e:
                logger.error(f"Unexpected error loading BM25 index: {e}", exc_info=True)
                bm25_index = None

        logger.info("Data loading attempt finished.")
        return metadata, embeddings, bm25_index