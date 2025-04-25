# -*- coding: utf-8 -*-
"""
Interface for interacting with different Large Language Model APIs.
Handles requests sequentially through configured providers and their models
(e.g., Google Model 1 -> Google Model 2 -> Groq Model 1 -> Groq Model 2).
Includes fallback logic for rate limiting or other specified errors.
"""

import logging
import time # Import time for potential delays between fallbacks
from .. import config # Use relative import for configuration
from typing import Optional, Union, Generator, List, Dict, Any, Tuple # Added Tuple

# --- Provider Specific Imports and Exception Handling ---
# Google Gemini
try:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted as GoogleResourceExhausted
    from google.api_core.exceptions import GoogleAPICallError, InvalidArgument
    GOOGLE_AVAILABLE = True
except ImportError:
    logging.getLogger(__name__).warning("Google Generative AI SDK not found (`pip install google-generativeai`). Google provider disabled.")
    GOOGLE_AVAILABLE = False
    # Define dummy exceptions if library not found to prevent NameErrors later
    class GoogleResourceExhausted(Exception): pass
    class GoogleAPICallError(Exception): pass
    class InvalidArgument(Exception): pass

# Groq
try:
    from groq import Groq, RateLimitError as GroqRateLimitError, APIError as GroqAPIError
    GROQ_AVAILABLE = True
except ImportError:
    logging.getLogger(__name__).warning("Groq SDK not found (`pip install groq`). Groq provider disabled.")
    GROQ_AVAILABLE = False
    # Define dummy exceptions
    class GroqRateLimitError(Exception): pass
    class GroqAPIError(Exception): pass

logger = logging.getLogger(__name__)

# --- Helper Function to Get All Attempt Configurations ---
def _get_llm_attempts() -> List[Dict[str, Any]]:
    """
    Generates a flattened list of LLM attempts based on provider order and models per provider.

    Returns:
        A list of dictionaries, each representing one attempt:
        {'provider': str, 'model_id': str, 'api_key': str | None}
    """
    attempts = []
    provider_details = {
        "google": {"sdk": GOOGLE_AVAILABLE, "models": config.LLM_GOOGLE_MODELS, "key": config.GOOGLE_API_KEY},
        "groq": {"sdk": GROQ_AVAILABLE, "models": config.LLM_GROQ_MODELS, "key": config.GROQ_API_KEY},
        # Add other providers here if needed
    }

    for provider_name in config.LLM_PROVIDER_ORDER:
        details = provider_details.get(provider_name)
        if not details:
            logger.warning(f"Provider '{provider_name}' in LLM_PROVIDER_ORDER is not recognized or supported.")
            continue

        if not details["sdk"]:
            logger.warning(f"SDK for provider '{provider_name}' not available. Skipping.")
            continue

        if not details["key"]:
            logger.warning(f"API key for provider '{provider_name}' not found. Skipping.")
            continue

        if not details["models"]:
            logger.warning(f"Model list for provider '{provider_name}' is empty. Skipping.")
            continue

        # Add attempts for each model within this provider
        for model_id in details["models"]:
            if model_id: # Ensure model_id is not empty
                attempts.append({
                    "provider": provider_name,
                    "model_id": model_id,
                    "api_key": details["key"]
                })

    if not attempts:
         logger.error("No valid LLM attempts could be configured (check provider order, model lists, keys, and SDKs).")

    return attempts


# --- Provider Specific API Call Functions (Streaming - Unchanged) ---
# _call_google_api_stream and _call_groq_api_stream remain the same as before
# They accept api_key, model_id, prompt, gen_args

def _call_google_api_stream(api_key: str, model_id: str, prompt: str, gen_args: Dict[str, Any]) -> Generator[str, None, None]:
    """Calls Google Gemini API and yields streaming response chunks."""
    # (Implementation from previous version - no changes needed here)
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_id)
        logger.info(f"Sending STREAMING request to Google API (Model: {model_id})...")
        # Adjust generation_config based on passed gen_args
        final_gen_args = gen_args.copy() # Avoid modifying original dict
        if 'generation_config' not in final_gen_args: # Ensure generation_config exists
             final_gen_args['generation_config'] = genai.types.GenerationConfig()
        # Apply max_output_tokens and temperature from gen_args if present
        if 'max_output_tokens' in final_gen_args:
             final_gen_args['generation_config'].max_output_tokens = final_gen_args.pop('max_output_tokens')
        if 'temperature' in final_gen_args:
             final_gen_args['generation_config'].temperature = final_gen_args.pop('temperature')

        response_stream = model.generate_content(prompt, stream=True, **final_gen_args)

        for chunk in response_stream:
            if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                 block_reason = chunk.prompt_feedback.block_reason
                 logger.warning(f"Google stream blocked: {block_reason}")
                 raise ValueError(f"Stream blocked by safety filters: {block_reason}") # Raise error
            text_part = getattr(chunk, 'text', None)
            if text_part:
                yield text_part
        logger.info(f"Google API stream finished successfully for model: {model_id}.")

    except InvalidArgument as e:
         logger.error(f"Google API Invalid Argument (check model ID '{model_id}' or prompt): {e}", exc_info=True)
         raise
    except GoogleAPICallError as e:
         logger.error(f"Google API call error for model {model_id}: {e}", exc_info=True)
         raise
    except Exception as e:
         logger.error(f"Unexpected error during Google streaming call for model {model_id}: {e}", exc_info=True)
         raise


def _call_groq_api_stream(api_key: str, model_id: str, prompt: str, gen_args: Dict[str, Any]) -> Generator[str, None, None]:
    """Calls Groq API and yields streaming response chunks."""
    # (Implementation from previous version - no changes needed here)
    try:
        client = Groq(api_key=api_key, timeout=config.LLM_API_TIMEOUT)
        logger.info(f"Sending STREAMING request to Groq API (Model: {model_id})...")
        stream = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_id,
            max_tokens=gen_args.get("max_output_tokens"),
            temperature=gen_args.get("temperature"),
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
        logger.info(f"Groq API stream finished successfully for model: {model_id}.")

    except GroqAPIError as e:
        logger.error(f"Groq API Error (Status {getattr(e, 'status_code', 'N/A')}) for model {model_id}: {getattr(e, 'message', e)}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during Groq streaming call for model {model_id}: {e}", exc_info=True)
        raise

# --- Main LLM Dispatcher Function (Streaming) ---
def get_llm_response_stream(prompt: str, generation_args: Optional[Dict[str, Any]] = None) -> Generator[str, None, None]:
    """
    Gets a streaming response from the configured LLM provider and model sequence.
    Iterates through providers defined in config.LLM_PROVIDER_ORDER and models within
    each provider (e.g., LLM_GOOGLE_MODELS, LLM_GROQ_MODELS). Falls back on specific errors.

    Args:
        prompt: The text prompt to send to the LLM.
        generation_args: Optional dictionary of additional arguments for the API call.
                         These are passed to the provider-specific call functions.

    Yields:
        str: Chunks of text from the LLM response. Yields error messages if failures occur.
    """
    llm_attempts = _get_llm_attempts()
    if not llm_attempts:
        yield "[Error: No available LLM attempts configured (check config.py and .env)]"
        return

    last_exception = None
    stream_successful = False

    # Prepare base generation args from config (can be overridden by generation_args)
    base_gen_args = {
        "max_output_tokens": config.LLM_MAX_NEW_TOKENS,
        "temperature": config.LLM_TEMPERATURE
    }
    if generation_args:
        base_gen_args.update(generation_args) # Merge/override with user args

    for attempt in llm_attempts:
        provider = attempt["provider"]
        model_id = attempt["model_id"]
        api_key = attempt["api_key"]

        # logger.info(f"Attempting LLM stream: Provider='{provider}', Model='{model_id}'") # Logged within specific call funcs now

        try:
            # Select the appropriate call function based on provider
            if provider == "google":
                response_generator = _call_google_api_stream(api_key, model_id, prompt, base_gen_args)
            elif provider == "groq":
                response_generator = _call_groq_api_stream(api_key, model_id, prompt, base_gen_args)
            else:
                # Should not happen if _get_llm_attempts filters correctly
                logger.error(f"Unsupported provider '{provider}' encountered in attempt loop.")
                last_exception = ValueError(f"Unsupported provider: {provider}")
                continue # Try next attempt

            # Consume the generator from the specific provider call
            chunk_yielded = False
            for chunk in response_generator:
                yield chunk
                chunk_yielded = True
                stream_successful = True # Mark success if at least one chunk yields

            # If we successfully yielded chunks, we are done.
            if stream_successful:
                return # Exit the main generator function successfully

            # Handle case where stream finished without yielding (and didn't raise below)
            if not chunk_yielded:
                 logger.warning(f"Stream from {provider} model {model_id} finished without yielding text.")
                 last_exception = ValueError(f"Empty stream from {provider} model {model_id}")
                 # Continue to the next attempt

        # --- Handle Specific Errors for Fallback ---
        except (GoogleResourceExhausted, GroqRateLimitError) as e:
            logger.warning(f"Rate limit hit for {provider} model {model_id}. Trying next attempt. Error: {type(e).__name__}")
            last_exception = e
            time.sleep(0.5) # Small delay before next attempt
            continue # Move to the next attempt in the list

        except ValueError as e: # Catch safety blocks or empty stream errors
             if "blocked by safety filters" in str(e) or "Empty stream" in str(e):
                  logger.warning(f"{str(e)} for {provider} model {model_id}. Trying next attempt.")
                  last_exception = e
                  continue # Move to the next attempt
             else: # Re-raise other ValueErrors
                  logger.error(f"ValueError during {provider} stream for model {model_id}: {e}", exc_info=True)
                  last_exception = e
                  break # Stop on unexpected ValueErrors

        except (GoogleAPICallError, GroqAPIError, InvalidArgument) as e:
             # Catch other API errors that likely won't be solved by switching models/providers
             logger.error(f"API Error for {provider} model {model_id} that prevents fallback: {type(e).__name__}", exc_info=True)
             last_exception = e
             break # Stop attempts

        except Exception as e:
            # Catch any other unexpected errors during the provider call
            logger.error(f"Unexpected error during {provider} stream for model {model_id}: {e}", exc_info=True)
            last_exception = e
            break # Stop attempts

    # If loop completes without successfully returning
    if not stream_successful:
        logger.error(f"All LLM streaming attempts failed. Last error: {last_exception}")
        yield f"[Error: All LLM attempts failed. Last error: {type(last_exception).__name__ if last_exception else 'N/A'}]"


# --- Non-Streaming Function ---
# Uses the updated streaming function internally
def get_llm_response(prompt: str, generation_args: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Gets a standard, non-streaming response from the configured LLM provider sequence.
    Uses the streaming function internally and collects the response.
    Args:
        prompt: The text prompt to send to the LLM.
        generation_args: Optional dictionary of additional arguments for the API call.
    Returns:
        The LLM's response text as a string, or None if all attempts fail or errors occur.
    """
    # Removed redundant log message - detailed logging happens in get_llm_response_stream
    # logger.info("Initiating non-streaming LLM request (will use streaming backend)...")
    full_response_parts = []
    error_occurred = False
    try:
        for chunk in get_llm_response_stream(prompt, generation_args):
            if chunk.startswith("[Error:"):
                logger.error(f"Error received from streaming backend: {chunk}")
                error_occurred = True
                # Stop collecting if a definitive error message is received
                full_response_parts = [chunk] # Store only the error message
                break
            full_response_parts.append(chunk)

        if error_occurred:
             logger.error("Non-streaming call failed due to errors in the stream.")
             return None # Return None on error
        elif not full_response_parts:
             logger.warning("Non-streaming call resulted in empty response from stream.")
             return None # Or "" depending on desired behavior
        else:
             final_response = "".join(full_response_parts).strip()
             logger.info("Non-streaming LLM request completed successfully.")
             return final_response

    except Exception as e:
        # Catch any unexpected errors during stream consumption
        logger.error(f"Unexpected error consuming LLM stream for non-streaming response: {e}", exc_info=True)
        return None
