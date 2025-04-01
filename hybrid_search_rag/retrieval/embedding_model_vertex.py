# hybrid_search_rag/retrieval/embedding_model_vertex.py
# --- NOTE: Renamed file to reflect Vertex AI implementation ---
"""Handles encoding text using the Google Vertex AI Embeddings API."""

import logging
import time
from typing import List, Union, Optional
import numpy as np
import math

# Import Google Cloud libraries
try:
    from google.cloud import aiplatform
    # TextEmbeddingModel is often found here, adjust if library structure changes
    from google.cloud.aiplatform.preview.language_models import TextEmbeddingModel
    # Optional: For specific exception handling
    # from google.api_core import exceptions as google_exceptions
except ImportError:
    # Provide specific instructions if the import fails
    raise ImportError(
        "Please install google-cloud-aiplatform: pip install google-cloud-aiplatform"
        "\nAlso ensure you have authenticated (e.g., `gcloud auth application-default login`)"
        )

# Assuming config is accessible via relative import
# Adjust relative import path if this file moves relative to config.py
from .. import config

logger = logging.getLogger(__name__)

# --- Configuration ---
# Get Vertex AI settings from config, providing defaults or raising errors
try:
    VERTEX_AI_PROJECT = config.VERTEX_AI_PROJECT
except AttributeError:
    raise AttributeError("VERTEX_AI_PROJECT not found in config. Please set it in hybrid_search_rag/config.py")

VERTEX_AI_LOCATION = getattr(config, 'VERTEX_AI_LOCATION', 'us-central1') # Default location if not set
# This MUST be a Vertex AI model ID now, read from config
VERTEX_AI_MODEL_ID = config.EMBEDDING_MODEL_NAME

# Vertex AI limits (check documentation for the specific model)
# Defaulting to 250 which is common for gecko, but allow override via config
VERTEX_AI_BATCH_LIMIT = getattr(config, 'VERTEX_AI_BATCH_LIMIT', 250)
# Delay between batches (optional)
API_DELAY_SECONDS = getattr(config, 'EMBEDDING_API_DELAY', 0.1) # Small default delay


class EmbeddingModel:
    """Wraps the Google Vertex AI API for encoding text."""

    _model_instance = None # Class-level cache for the model client instance

    def __init__(self, model_name: str):
        """
        Initializes the EmbeddingModel wrapper for Vertex AI.
        Ensures Vertex AI client is initialized and the specified model is loaded.
        Args:
            model_name (str): The name of the embedding model (should match config.EMBEDDING_MODEL_NAME).
        """
        # Check if the model name passed matches the one configured for Vertex AI
        if model_name != VERTEX_AI_MODEL_ID:
            logger.warning(f"EmbeddingModel initialized with name '{model_name}', "
                           f"but API calls will use Vertex AI model '{VERTEX_AI_MODEL_ID}' from config.")
            # Depending on strictness, you might want to raise an error here

        self.model_name = VERTEX_AI_MODEL_ID # Store the actual model ID being used

        # Initialize Vertex AI client if not already done
        try:
            # Check if already initialized (simple check)
            # A more robust check might involve trying a simple API call or checking internal state
            if not aiplatform.constants.global_config.initialized:
                 logger.info(f"Initializing Vertex AI client for project='{VERTEX_AI_PROJECT}', location='{VERTEX_AI_LOCATION}'...")
                 aiplatform.init(project=VERTEX_AI_PROJECT, location=VERTEX_AI_LOCATION)
                 logger.info("Vertex AI client initialized successfully.")
            else:
                 logger.info("Vertex AI client already initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI client: {e}", exc_info=True)
            # Provide more context for common auth errors
            if "Application Default Credentials" in str(e):
                 logger.error("Hint: Ensure you have authenticated via `gcloud auth application-default login` or set up a service account.")
            raise RuntimeError(f"Vertex AI Initialization Failed: {e}")

        # Load or get the cached model instance
        if EmbeddingModel._model_instance is None:
             try:
                 logger.info(f"Loading Vertex AI embedding model: {self.model_name}")
                 # Ensure the class TextEmbeddingModel is available
                 if 'TextEmbeddingModel' not in globals():
                      # Attempt import again if it failed initially but google.cloud.aiplatform succeeded
                      from google.cloud.aiplatform.preview.language_models import TextEmbeddingModel
                 EmbeddingModel._model_instance = TextEmbeddingModel.from_pretrained(self.model_name)
                 logger.info(f"Vertex AI model '{self.model_name}' loaded.")
             except Exception as e:
                 logger.error(f"Failed to load Vertex AI model '{self.model_name}': {e}", exc_info=True)
                 raise RuntimeError(f"Vertex AI Model Loading Failed: {e}")

        # Assign the loaded/cached model instance to this object
        self.model = EmbeddingModel._model_instance


    def encode(self, texts: Union[str, List[str]], batch_size: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Encodes text(s) into vector embeddings using the Vertex AI API with batching.
        Uses the configured VERTEX_AI_BATCH_LIMIT.

        Args:
            texts (Union[str, List[str]]): A single string or a list of strings to encode.
            batch_size (Optional[int]): This parameter is ignored; uses VERTEX_AI_BATCH_LIMIT.

        Returns:
            Optional[np.ndarray]: A NumPy array of embeddings (shape: num_texts, embedding_dim),
                                 or None if encoding failed.
        """
        if self.model is None:
             logger.error("Vertex AI embedding model instance is not available for encoding.")
             return None

        if batch_size is not None:
             logger.warning(f"Ignoring passed batch_size ({batch_size}), using Vertex AI limit: {VERTEX_AI_BATCH_LIMIT}")

        # --- Input Validation and Preparation ---
        if isinstance(texts, str):
            texts_list = [texts] # Treat single string as a list of one
        elif isinstance(texts, list):
            # Ensure all items in the list are strings, replace None with empty string
            texts_list = [str(t) if t is not None else "" for t in texts]
        else:
            # Attempt to convert other types to a single string in a list
            texts_list = [str(texts)]

        # Filter out any potentially empty strings *after* conversion if the API requires non-empty inputs
        # Note: Check Vertex AI docs if empty strings are allowed or cause errors. Assuming they are allowed for now.
        # texts_list = [t for t in texts_list if t.strip()] # Optional: remove empty strings

        if not texts_list: # Check if list is empty after potential filtering
            logger.warning("Input contains no valid text to encode after validation/filtering.")
            return np.array([], dtype=np.float32) # Return empty array

        all_embedding_values = [] # To store the resulting embedding vectors (lists of floats)
        num_texts = len(texts_list)
        # Use the Vertex AI specific batch limit for processing
        effective_batch_size = VERTEX_AI_BATCH_LIMIT
        num_batches = math.ceil(num_texts / effective_batch_size)

        logger.info(f"Encoding {num_texts} texts using Vertex AI model '{self.model_name}' "
                    f"in {num_batches} batches (API limit {effective_batch_size}, delay {API_DELAY_SECONDS}s).")

        for i in range(num_batches):
            batch_start = i * effective_batch_size
            batch_end = min((i + 1) * effective_batch_size, num_texts)
            batch_texts = texts_list[batch_start:batch_end]

            # Skip if batch is somehow empty
            if not batch_texts:
                logger.warning(f"Skipping empty batch {i+1}/{num_batches}.")
                continue

            logger.debug(f"Sending batch {i+1}/{num_batches} ({len(batch_texts)} items) to Vertex AI API...")

            try:
                # --- Make the API Call ---
                # The get_embeddings method handles interaction with the Vertex AI endpoint.
                # It accepts a list of strings.
                response_embeddings = self.model.get_embeddings(batch_texts)

                # --- Process the Response ---
                # response_embeddings should be a list of TextEmbedding objects.
                # Each object has a .values attribute containing the embedding vector (list of floats).
                if not isinstance(response_embeddings, list) or not all(hasattr(emb, 'values') for emb in response_embeddings):
                     logger.error(f"Vertex AI API returned unexpected response format for batch {i+1}. Expected list of TextEmbedding objects.")
                     # Log part of the response for debugging if possible
                     try: logger.error(f"Response sample: {response_embeddings[:2]}")
                     except: pass
                     return None # Stop processing on unexpected format

                batch_embedding_values = [emb.values for emb in response_embeddings]

                # Basic validation of embedding structure (check first one)
                if batch_embedding_values and not isinstance(batch_embedding_values[0], list):
                     logger.error(f"Embeddings in batch {i+1} do not appear to be lists of numbers.")
                     return None

                all_embedding_values.extend(batch_embedding_values)
                logger.debug(f"Batch {i+1}/{num_batches} successful. Received {len(batch_embedding_values)} embeddings.")

                # --- Optional Delay ---
                # Add delay between batches if configured, useful for very high throughput scenarios
                # although often not strictly necessary for Vertex AI quotas compared to free tiers.
                if i < num_batches - 1 and API_DELAY_SECONDS > 0:
                    logger.debug(f"Waiting {API_DELAY_SECONDS}s before next batch...")
                    time.sleep(API_DELAY_SECONDS)

            # --- Error Handling ---
            # Catch more specific Google Cloud errors if needed, e.g.,
            # except google_exceptions.ResourceExhaustedError as e:
            #     logger.error(f"Vertex AI Quota Error (Resource Exhausted) on batch {i+1}: {e}")
            #     # Implement more sophisticated retry or stop logic here
            #     return None
            # except google_exceptions.GoogleAPICallError as e:
            #     logger.error(f"Vertex AI API Call Error on batch {i+1}: {e}")
            #     return None
            except Exception as e:
                logger.error(f"An unexpected error occurred during Vertex AI encoding batch {i+1}: {e}", exc_info=True)
                # Decide if processing should stop or continue
                return None # Stop processing on any batch error for now

        # --- Final Conversion and Return ---
        if len(all_embedding_values) == num_texts:
            logger.info(f"Successfully encoded all {num_texts} texts using Vertex AI.")
            try:
                # Convert list of lists (embeddings) to a NumPy array
                embeddings_np = np.array(all_embedding_values, dtype=np.float32)
                logger.info(f"Final embedding shape: {embeddings_np.shape}")
                # Final check: ensure embedding dimension is consistent
                if embeddings_np.ndim == 2 and embeddings_np.shape[0] == num_texts:
                     return embeddings_np
                else:
                     logger.error(f"Final NumPy array shape is incorrect: {embeddings_np.shape}. Expected ({num_texts}, embedding_dim).")
                     return None
            except ValueError as e:
                 # Error during np.array conversion (e.g., inconsistent embedding lengths)
                 logger.error(f"Failed to convert collected embeddings to NumPy array: {e}. Check embedding consistency.")
                 if all_embedding_values: logger.error(f"Example embedding lengths: {[len(e) for e in all_embedding_values[:3]]}")
                 return None
        else:
            # If the number of collected embeddings doesn't match the input text count
            logger.error(f"Encoding failed: Expected {num_texts} embeddings, but received {len(all_embedding_values)}.")
            return None
