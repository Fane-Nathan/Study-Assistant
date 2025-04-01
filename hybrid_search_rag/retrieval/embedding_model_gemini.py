# hybrid_search_rag/retrieval/embedding_model_gemini.py
"""Handles encoding text using the Google Generative AI (Gemini) API."""

import logging
import time
from typing import List, Union, Optional
import numpy as np
import math
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Assuming config is accessible via relative import
from .. import config

logger = logging.getLogger(__name__)

# --- Configuration ---
# Get API Key from config
try:
    GOOGLE_API_KEY = config.GOOGLE_API_KEY
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set in config or .env file.")
except (AttributeError, ValueError) as e:
    logger.error(f"Configuration Error: {e}")
    # Decide how to handle this - raise error or allow to proceed and fail later?
    # For now, log error and let it potentially fail during initialization.
    GOOGLE_API_KEY = None

# Define the Gemini embedding model to use (check Google AI documentation for latest/best)
# Common choice for general purpose embeddings accessible via basic API key.
DEFAULT_GEMINI_EMBEDDING_MODEL = "models/embedding-001"
# Allow override from config if needed, otherwise use default
GEMINI_EMBEDDING_MODEL_ID = getattr(config, 'GEMINI_EMBEDDING_MODEL', DEFAULT_GEMINI_EMBEDDING_MODEL)

# Batching and Delay (adjust based on observed API behavior/limits)
GEMINI_API_BATCH_LIMIT = 100 # Gemini's embed_content can take a list, limit might be total size or count. 100 is a reasonable starting point.
API_DELAY_SECONDS = getattr(config, 'EMBEDDING_API_DELAY', 0.1) # Small default delay


# Define exceptions that might warrant retries (e.g., rate limits)
# This requires knowing the specific exception types google.generativeai might raise.
# Example (might need adjustment):
# from google.api_core import exceptions as google_exceptions
# RETRYABLE_EXCEPTIONS = (google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable)
# For now, using a broader retry on Exception, which is less ideal.
RETRYABLE_EXCEPTIONS = (Exception,) # Less specific, retries on more errors

class EmbeddingModel:
    """Wraps the Google Generative AI (Gemini) API for encoding text."""

    _is_configured = False # Class-level flag for API key configuration

    def __init__(self, model_name: str = GEMINI_EMBEDDING_MODEL_ID):
        """
        Initializes the EmbeddingModel wrapper for the Gemini API.
        Configures the genai library with the API key.

        Args:
            model_name (str): The name of the Gemini embedding model to use.
        """
        self.model_name = model_name
        logger.info(f"Initializing EmbeddingModel for Gemini API with model: {self.model_name}")

        if not GOOGLE_API_KEY:
            # Logged error earlier, raise runtime error if key is missing
            raise RuntimeError("GOOGLE_API_KEY not found. Cannot initialize Gemini EmbeddingModel.")

        # Configure the genai library only once per session/class load
        if not EmbeddingModel._is_configured:
            try:
                logger.info("Configuring Google Generative AI library...")
                genai.configure(api_key=GOOGLE_API_KEY)
                EmbeddingModel._is_configured = True
                logger.info("Google Generative AI library configured successfully.")
            except Exception as e:
                logger.error(f"Failed to configure Google Generative AI library: {e}", exc_info=True)
                # Prevent further attempts if configuration fails
                EmbeddingModel._is_configured = False # Ensure it remains false
                raise RuntimeError(f"Google Generative AI Configuration Failed: {e}")
        else:
             logger.info("Google Generative AI library already configured.")


    # Use tenacity for automatic retries on specific errors (e.g., rate limits)
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS) # Adjust exception types if needed
    )
    def _embed_batch(self, batch_texts: List[str], task_type: str) -> Optional[List[List[float]]]:
        """Internal method to embed a single batch with retries."""
        try:
            logger.debug(f"Sending batch of {len(batch_texts)} items to Gemini API (model: {self.model_name}, task: {task_type})...")
            # Make the API call using genai.embed_content
            result = genai.embed_content(
                model=self.model_name,
                content=batch_texts,
                task_type=task_type
            )

            # Process the response
            if isinstance(result, dict) and 'embedding' in result and isinstance(result['embedding'], list):
                # Check if the number of embeddings matches the input batch size
                if len(result['embedding']) == len(batch_texts):
                    logger.debug(f"Batch successful. Received {len(result['embedding'])} embeddings.")
                    # Assuming each element in result['embedding'] is a list of floats
                    return result['embedding']
                else:
                    logger.error(f"Gemini API returned {len(result['embedding'])} embeddings for a batch of {len(batch_texts)}.")
                    return None
            else:
                logger.error(f"Gemini API returned unexpected response format for batch. Response: {result}")
                return None

        except Exception as e:
            # Log the error before tenacity retries or gives up
            logger.error(f"Error during Gemini embed_content call: {e}", exc_info=True)
            # Re-raise the exception so tenacity can handle retries
            raise e

    def encode(self,
               texts: Union[str, List[str]],
               batch_size: Optional[int] = None,
               task_type: str = "RETRIEVAL_DOCUMENT" # Default task type
               ) -> Optional[np.ndarray]:
        """
        Encodes text(s) into vector embeddings using the Gemini API with batching.

        Args:
            texts (Union[str, List[str]]): A single string or a list of strings to encode.
            batch_size (Optional[int]): If None, uses GEMINI_API_BATCH_LIMIT.
            task_type (str): The task type for the embedding ('RETRIEVAL_DOCUMENT',
                             'RETRIEVAL_QUERY', 'SEMANTIC_SIMILARITY', etc.).

        Returns:
            Optional[np.ndarray]: A NumPy array of embeddings (shape: num_texts, embedding_dim),
                                  or None if encoding failed.
        """
        if not EmbeddingModel._is_configured:
             logger.error("Gemini API not configured. Cannot encode.")
             return None

        # --- Input Validation and Preparation ---
        if isinstance(texts, str):
            texts_list = [texts]
        elif isinstance(texts, list):
            texts_list = [str(t) if t is not None else "" for t in texts]
        else:
            texts_list = [str(texts)]

        # Optional: remove empty strings if they cause issues
        # texts_list = [t for t in texts_list if t.strip()]

        if not texts_list:
            logger.warning("Input contains no valid text to encode after validation/filtering.")
            return np.array([], dtype=np.float32)

        all_embedding_values: List[List[float]] = []
        num_texts = len(texts_list)
        effective_batch_size = batch_size if batch_size is not None else GEMINI_API_BATCH_LIMIT
        num_batches = math.ceil(num_texts / effective_batch_size)

        logger.info(f"Encoding {num_texts} texts using Gemini model '{self.model_name}' "
                    f"in {num_batches} batches (size {effective_batch_size}, task {task_type}, delay {API_DELAY_SECONDS}s).")

        for i in range(num_batches):
            batch_start = i * effective_batch_size
            batch_end = min((i + 1) * effective_batch_size, num_texts)
            batch_texts = texts_list[batch_start:batch_end]

            if not batch_texts:
                logger.warning(f"Skipping empty batch {i+1}/{num_batches}.")
                continue

            # Call the internal batch embedding method with retries
            batch_embeddings = self._embed_batch(batch_texts, task_type)

            if batch_embeddings is None:
                logger.error(f"Failed to embed batch {i+1}/{num_batches} after retries. Aborting encoding.")
                return None # Stop processing if a batch fails

            all_embedding_values.extend(batch_embeddings)

            # Optional Delay
            if i < num_batches - 1 and API_DELAY_SECONDS > 0:
                logger.debug(f"Waiting {API_DELAY_SECONDS}s before next batch...")
                time.sleep(API_DELAY_SECONDS)

        # --- Final Conversion and Return ---
        if len(all_embedding_values) == num_texts:
            logger.info(f"Successfully encoded all {num_texts} texts using Gemini API.")
            try:
                embeddings_np = np.array(all_embedding_values, dtype=np.float32)
                if embeddings_np.ndim == 2 and embeddings_np.shape[0] == num_texts:
                     logger.info(f"Final embedding shape: {embeddings_np.shape}")
                     return embeddings_np
                else:
                     logger.error(f"Final NumPy array shape is incorrect: {embeddings_np.shape}. Expected ({num_texts}, embedding_dim).")
                     # Log lengths for debugging
                     if all_embedding_values: logger.error(f"Sample embedding lengths: {[len(e) for e in all_embedding_values[:3]]}")
                     return None
            except ValueError as e:
                 logger.error(f"Failed to convert collected embeddings to NumPy array: {e}. Check embedding consistency.")
                 if all_embedding_values: logger.error(f"Sample embedding lengths: {[len(e) for e in all_embedding_values[:3]]}")
                 return None
        else:
            logger.error(f"Encoding mismatch: Expected {num_texts} embeddings, but received {len(all_embedding_values)}.")
            return None