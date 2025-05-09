# hybrid_search_rag/retrieval/embedding_model_gemini.py
"""Handles encoding text using the Google Generative AI (Gemini) API."""

import logging
import time
from typing import List, Union, Optional
import numpy as np
import math
import google.generativeai as genai
from tenacity import stop_after_attempt, wait_exponential, retry_if_exception_type, stop, wait, retry

# Assuming config is accessible via relative import
from .. import config

logger = logging.getLogger(__name__)

# --- Configuration ---
try:
    GOOGLE_API_KEY = config.GOOGLE_API_KEY
    if not GOOGLE_API_KEY:        
        raise RuntimeError("GOOGLE_API_KEY not set in config or .env file.")    
except (AttributeError, ValueError) as e:
    raise RuntimeError(f"Configuration Error: {e}. Cannot initialize Gemini EmbeddingModel.") from e

# Use default model ID or override from config
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
GEMINI_EMBEDDING_MODEL_ID = getattr(config, 'GEMINI_EMBEDDING_MODEL', GEMINI_EMBEDDING_MODEL)

# Batching and Delay settings from config or defaults
GEMINI_API_BATCH_LIMIT = getattr(config, 'GEMINI_API_BATCH_LIMIT', 100) # Default batch limit
API_DELAY_SECONDS = getattr(config, 'EMBEDDING_API_DELAY', 0.1) # Default delay

# Define exceptions that might warrant retries (e.g., rate limits, temporary server errors)
try:
    from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
    RETRYABLE_EXCEPTIONS = (ResourceExhausted, ServiceUnavailable)
except ImportError:
    logger.warning("google-api-core not installed. Using broad Exception for retries.")
    RETRYABLE_EXCEPTIONS = (Exception,)
    # if necessary, refine with specific Google API errors


def retry_with_exponential_backoff(
    stop: stop.stop_base = stop_after_attempt(3),
    wait: wait.wait_base = wait_exponential(multiplier=1, min=2, max=10),
    retry_condition = retry_if_exception_type(RETRYABLE_EXCEPTIONS)
):
    """Applies exponential backoff retry logic to API calls.
    The parameters control the retry behavior (stop condition, wait strategy, retry condition)."""
    """
    Decorator to retry a function with exponential backoff.

    Args:
        stop: Stop condition (e.g., after a maximum number of attempts)
        wait: Wait strategy (e.g., exponential backoff)
        retry: Retry condition (e.g., based on exception type).

    Returns:
        Decorated function.
    """
    def decorator(func):
        return retry(stop=stop, wait=wait, retry=retry_condition)(func)
    return decorator

class EmbeddingModel:
    """Wraps the Google Generative AI (Gemini) API for encoding text."""

    _is_configured = False # Class flag to track if genai.configure has been called

    def __init__(self, model_name: str = GEMINI_EMBEDDING_MODEL_ID):
        """Initializes the wrapper and configures the genai library if needed."""
        self.model_name = model_name
        logger.info(f"Initializing EmbeddingModel for Gemini API (model: {self.model_name})")

        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY not found. Cannot initialize Gemini EmbeddingModel.")

        # Configure the genai library only once per class load
        if not EmbeddingModel._is_configured:
            try:
                logger.info("Configuring Google Generative AI library...")
                genai.configure(api_key=GOOGLE_API_KEY)
                EmbeddingModel._is_configured = True
                logger.info("Google Generative AI library configured.")
            except Exception as e:
                logger.error(f"Failed to configure Google Generative AI library: {e}", exc_info=True)
                EmbeddingModel._is_configured = False # Ensure flag remains false on error
                raise RuntimeError(f"Google Generative AI Configuration Failed: {e}")
        # else: # No need to log if already configured unless verbose
        #     logger.info("Google Generative AI library already configured.")

    # Use tenacity for automatic retries on potentially transient API errors
    @retry_with_exponential_backoff()
    def _embed_batch(self, batch_texts: List[str], task_type: str) -> Optional[List[List[float]]]:
        """Handles the actual API call for embedding a batch of texts,
        including error handling and retries."""
        try:
            logger.debug(f"Sending batch ({len(batch_texts)} items) to Gemini API...")  # Debug level
            # Call the Gemini embedding API
            result = genai.embed_content(
                model=self.model_name,
                content=batch_texts,
                task_type=task_type # Important: e.g., 'RETRIEVAL_DOCUMENT' or 'RETRIEVAL_QUERY'
            )

            # Validate and extract embeddings from the response dictionary
            if isinstance(result, dict) and 'embedding' in result and isinstance(result['embedding'], list):
                if len(result['embedding']) == len(batch_texts):
                   
                    return result['embedding'] # Return list of embedding lists (floats)
                else:
                    # Mismatch indicates an API issue or unexpected response format
                    logger.error(f"Gemini API returned {len(result['embedding'])} embeddings for a batch of {len(batch_texts)}.")
                    return None
            else:
                logger.error(f"Gemini API returned unexpected response format: {result}")
                return None

        except Exception as e:
            # Log error before tenacity retries or gives up
            logger.error(f"Error during Gemini embed_content API call: {e}", exc_info=True)
            raise e # Re-raise exception to allow tenacity retry mechanism

    def encode(self,
               texts: Union[str, List[str]],               
               batch_size: Optional[int] = None,              
               task_type: str = "RETRIEVAL_DOCUMENT"              
               ) -> Optional[np.ndarray]:        
        """
        Encodes text(s) into vector embeddings using the Gemini API with batching and retries.

        Args:
            texts: A single string or a list of strings.
            batch_size: Max number of texts per API call (uses config default if None).
            task_type: Embedding task type ('RETRIEVAL_DOCUMENT', 'RETRIEVAL_QUERY', etc.).

        Returns:
            A NumPy array of embeddings, or None if encoding failed.
        """
        if not EmbeddingModel._is_configured:            
            logger.error("Gemini API not configured. Cannot encode.")
            return None

        # --- Input Validation --- # Check for valid input text.
        if isinstance(texts, str):
             texts_list = [texts]
             
        elif isinstance(texts, list): texts_list = [str(t) if t is not None else "" for t in texts]
        else: texts_list = [str(texts)] # Attempt conversion for other types

        # remove empty strings if they cause API issues
        original_len = len(texts_list)
        texts_list = [t for t in texts_list if t.strip()]
        removed_count = original_len - len(texts_list)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} empty string(s) from input. They will be skipped during encoding.")

        if not texts_list:
            logger.warning("Input contains no valid text to encode after validation.")
            return np.array([], dtype=np.float32) # Return empty array for empty input

        # --- Batch Processing --- # Split the input into batches for API calls.
        all_embedding_values: List[List[float]] = []
        num_texts = len(texts_list)
        logger.debug(f"Encoding {num_texts} texts before batching.")
        # Use provided batch_size or the configured limit
        effective_batch_size = batch_size if batch_size is not None else GEMINI_API_BATCH_LIMIT
        num_batches = math.ceil(num_texts / effective_batch_size)

        logger.info(f"Encoding {num_texts} texts using Gemini '{self.model_name}' in {num_batches} batches (size={effective_batch_size}, task={task_type}).")

        for i in range(num_batches):
            logger.debug(f"Processing batch {i + 1}/{num_batches} (batch_size={effective_batch_size}).")
            batch_start = i * effective_batch_size
            batch_end = min((i + 1) * effective_batch_size, num_texts)
            batch_texts = texts_list[batch_start:batch_end]
            logger.debug(f"Batch {i + 1} start index: {batch_start}, end index: {batch_end - 1}.")

            if not batch_texts: continue  # Skip empty batches

            # Call the internal method which includes retry logic
            batch_embeddings = self._embed_batch(batch_texts, task_type)

            if batch_embeddings is None:
                logger.error(f"Failed to embed batch {i+1}/{num_batches} after retries. Aborting encoding.")
                return None # Stop processing if any batch fails definitively

            logger.debug(f"Successfully embedded batch {i+1}/{num_batches}. Received {len(batch_embeddings)} embeddings.")
            all_embedding_values.extend(batch_embeddings)

            # Optional delay between batches (if API rate limits are tight) # Add delay between batches.
            if i < num_batches - 1 and API_DELAY_SECONDS > 0:
                logger.debug(f"Waiting {API_DELAY_SECONDS}s before next batch...")  # Debug level
                time.sleep(API_DELAY_SECONDS)
                 

                
        # --- Final Conversion and Validation --- # Convert results to NumPy array and check shape.
        if len(all_embedding_values) == num_texts:
            logger.info(f"Successfully encoded all {num_texts} texts.")
            try:
                embeddings_np = np.array(all_embedding_values, dtype=np.float32)
                # Final check on array shape
                if embeddings_np.ndim == 2 and embeddings_np.shape[0] == num_texts:
                     logger.info(f"Final embedding shape: {embeddings_np.shape}")
                     return embeddings_np
                else:
                     logger.error(f"Final NumPy array shape is incorrect: {embeddings_np.shape}. Expected ({num_texts}, embedding_dim).")
                     if all_embedding_values: logger.error(f"Sample embedding lengths: {[len(e) for e in all_embedding_values[:3]]}") # Log length hint
                     return None
            except ValueError as e:
                 # Error during np.array conversion, likely inconsistent embedding lengths from API?
                 logger.error(f"Failed to convert embeddings to NumPy array: {e}. Check embedding consistency.")
                 if all_embedding_values: logger.error(f"Sample embedding lengths: {[len(e) for e in all_embedding_values[:3]]}")
                 return None
        else:
            # Should not happen if batch processing logic is correct, but check anyway
            logger.error(f"Encoding mismatch: Expected {num_texts} embeddings, received {len(all_embedding_values)}.")
            return None