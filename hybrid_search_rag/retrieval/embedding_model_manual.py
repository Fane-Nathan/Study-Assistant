# hybrid_search_rag/retrieval/embedding_model.py
"""Handles loading the Sentence-BERT model and encoding text."""

from sentence_transformers import SentenceTransformer
import logging
from typing import List, Union, Optional
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Wraps the SentenceTransformer model for encoding text."""
    _instance = None # Singleton instance
    _model = None
    _model_name = None

    def __new__(cls, model_name: str):
        """Implement Singleton pattern to load model only once."""
        if cls._instance is None:
            logger.info("Creating new EmbeddingModel instance.")
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._model_name = model_name
            cls._model = None
            cls._instance._check_gpu() # Check GPU on first instance creation
        elif cls._model_name != model_name:
            logger.warning(f"EmbeddingModel already initialized with {cls._model_name}. Ignoring new name {model_name}.")
            # Optionally, could reload the model here if switching is desired, but complicates things.
        return cls._instance

    def __init__(self, model_name: str):
        """Initializes the EmbeddingModel attributes if not already set."""
        # Init is called even for singleton, ensure attributes are set
        if not hasattr(self, 'model_name') or self.model_name is None:
             self.model_name = model_name
        if not hasattr(self, 'model') or self.model is None:
            self.model = self._model # Use the class-level model


    def _check_gpu(self):
        """Checks for TensorFlow GPU availability and sets memory growth."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"TensorFlow found GPUs: {gpus}")
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info("Enabled memory growth for GPUs.")
                except RuntimeError as e:
                    logger.warning(f"Could not set memory growth (likely already initialized): {e}")
            else:
                logger.info("No GPU found by TensorFlow, using CPU for potential TF ops.")
        except Exception as e:
            logger.error(f"Error during GPU check: {e}")

    def load_model(self):
        """Loads the SentenceTransformer model if not already loaded."""
        if EmbeddingModel._model is None: # Check class variable
            if not EmbeddingModel._model_name:
                 logger.error("Cannot load model: model_name not set.")
                 return
            try:
                logger.info(f"Loading SentenceTransformer model: {EmbeddingModel._model_name}...")
                # sentence-transformers uses PyTorch by default unless TF is forced/only option
                # device=None should auto-detect best available (CPU/GPU/MPS)
                EmbeddingModel._model = SentenceTransformer(EmbeddingModel._model_name, device=None)
                self.model = EmbeddingModel._model # Update instance variable too
                # Store the device the model was loaded onto
                detected_device = EmbeddingModel._model.device
                # Store on class for singleton consistency and instance for direct access
                EmbeddingModel._device = detected_device
                self.device = detected_device
                logger.info(f"Model {EmbeddingModel._model_name} loaded successfully on device: {EmbeddingModel._model.device}")
            except Exception as e:
                logger.error(f"Failed to load embedding model '{EmbeddingModel._model_name}': {e}", exc_info=True)
                # Set model to None explicitly on failure
                EmbeddingModel._model = None
                self.model = None
                if hasattr(self, 'device'): self.device = None
                raise

    def get_model(self) -> Optional[SentenceTransformer]:
        """Returns the loaded model instance, loading it if necessary."""
        if self.model is None:
             self.load_model() # Attempt to load if not already loaded
        return self.model


    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> Optional[np.ndarray]:
        """Encodes text(s) into vector embeddings using the loaded model."""
        model_instance = self.get_model() # Ensure model is loaded
        if model_instance is None:
            logger.error("Embedding model is not available for encoding.")
            return None

        # Input sanitization
        if isinstance(texts, list):
            valid_texts = [str(t) if t is not None else "" for t in texts]
            # Check if all items are effectively empty after conversion
            if not any(t.strip() for t in valid_texts):
                logger.warning("Input list contains only empty or None strings after validation. Encoding empty strings.")
                # Fall through to encode empty strings if needed, or return early:
                # return np.array([]) # Or handle as appropriate
        elif isinstance(texts, str):
            valid_texts = texts
        elif texts is None:
             valid_texts = ""
        else:
            # Attempt conversion for other types
            valid_texts = str(texts)

        # Check single string case after potential conversion
        if not valid_texts and isinstance(valid_texts, str):
             logger.warning("Input text is empty or None. Encoding empty string.")
             # Fall through to encode empty string

        try:
            num_items = 1 if isinstance(valid_texts, str) else len(valid_texts)
            logger.info(f"Encoding {num_items} item(s) using {self.model_name}...")
            # Pass device explicitly? 'None' should work, but check SBERT docs if issues
            embeddings = model_instance.encode(
                texts,
                batch_size=batch_size,  # <--- TRY SMALLER VALUES (e.g., 16, 8, 4, or even 1)
                show_progress_bar=True, # Keep progress bar if you have it
                convert_to_tensor=False, # Usually False if you need numpy array
            )
            logger.info("Encoding complete.")
            # Ensure numpy array output
            return np.array(embeddings, dtype=np.float32) if not isinstance(embeddings, np.ndarray) else embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"An error occurred during text encoding: {e}", exc_info=True)
            return None