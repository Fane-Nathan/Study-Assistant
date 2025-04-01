# hybrid_search_rag/llm/llm_interface.py
"""Interface for interacting with different Large Language Model APIs.
Routes requests to the configured LLM provider (Groq, HF, Google)."""

import logging
import requests # Needed for Hugging Face
from groq import Groq, RateLimitError, APIError # Groq's specific library
from tenacity import retry, stop_after_attempt, wait_exponential # For robust API calls with retries
from .. import config # Use relative import for configuration
from typing import Union

# Import Google Generative AI library
try:
    import google.generativeai as genai
except ImportError:
    raise ImportError("Required package 'google-generativeai' not found. Please install it: pip install google-generativeai")

logger = logging.getLogger(__name__)

# --- Groq API Call Function ---
# Uses tenacity's @retry decorator for automatic retries on failure.
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def _call_groq_api(prompt: str) -> Union[str, None]:
    """Calls the Groq API using configured settings."""
    logger.info(f"Sending request to Groq API (Model: {config.LLM_MODEL_ID})...")

    if not config.GROQ_API_KEY:
        logger.error("Groq API Key not configured. Cannot call Groq API.")
        return None

    try:
        client = Groq(api_key=config.GROQ_API_KEY, timeout=config.LLM_API_TIMEOUT)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=config.LLM_MODEL_ID,
            max_tokens=config.LLM_MAX_NEW_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            stream=False, # Get full response at once
        )
        response_text = chat_completion.choices[0].message.content
        logger.info("Groq API request successful.")
        return response_text.strip()

    except RateLimitError as e:
        logger.error(f"Groq API Rate Limit Error: {e}. Check usage limits on console.groq.com.")
        raise e # Re-raise to allow tenacity retry
    except APIError as e:
        logger.error(f"Groq API Error (Status {e.status_code}): {e.message}")
        return None # Don't retry other API errors
    except Exception as e:
        logger.error(f"Unexpected error during Groq API call: {e}", exc_info=True)
        return None

# --- Hugging Face API Call Function ---
# Also uses retry for robustness.
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def _call_huggingface_api(prompt: str) -> Union[str, None]:
    """Calls the Hugging Face Inference API."""
    # Check for required HF-specific configuration
    if not hasattr(config, 'HF_MODEL_ID') or not hasattr(config, 'HF_API_URL'):
         logger.error("HF provider selected, but HF_MODEL_ID or HF_API_URL missing in config.py.")
         return None
    logger.info(f"Sending request to Hugging Face Inference API ({config.HF_MODEL_ID})...")

    if not config.HF_API_TOKEN or not config.HF_API_URL:
        logger.error("Hugging Face API Token or URL not configured.")
        return None

    headers = {"Authorization": f"Bearer {config.HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": config.LLM_MAX_NEW_TOKENS,
            # Set temperature to None if <= 0, as some HF models require > 0
            "temperature": config.LLM_TEMPERATURE if config.LLM_TEMPERATURE > 0 else None,
            "return_full_text": False, # Only get the generated part
        },
        "options": {"wait_for_model": True} # Wait if model is loading
    }
    # Remove temperature parameter if it was set to None
    if payload["parameters"]["temperature"] is None:
        del payload["parameters"]["temperature"]

    try:
        response = requests.post(
            config.HF_API_URL,
            headers=headers,
            json=payload,
            timeout=config.LLM_API_TIMEOUT
        )
        response.raise_for_status() # Check for HTTP 4xx/5xx errors
        result = response.json()

        # Extract generated text (handle potential variations in HF response format)
        generated_text = None
        if isinstance(result, list) and result:
            generated_text = result[0].get("generated_text")
        elif isinstance(result, dict):
             generated_text = result.get("generated_text")

        if generated_text:
            logger.info("HF API request successful.")
            return generated_text.strip()
        else:
            logger.error(f"HF API Error: Could not find 'generated_text' in response: {result}")
            return None

    except requests.exceptions.HTTPError as e:
        error_details = e.response.text
        logger.error(f"HF API request failed: Status {e.response.status_code}. Details: {error_details}")
        if e.response.status_code == 429: raise e # Re-raise for rate limit retries
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"HF API request failed (network/timeout): {e}", exc_info=True)
        raise e # Re-raise for retries on potentially transient errors
    except Exception as e:
        logger.error(f"Unexpected error during HF API call: {e}", exc_info=True)
        return None

# --- Google Generative AI (Gemini) API Call Function ---
# Also uses retry.
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def _call_google_api(prompt: str) -> Union[str, None]:
    """Calls the Google Generative AI API (Gemini models)."""
    logger.info(f"Sending request to Google API (Model: {config.LLM_MODEL_ID})...")

    if not config.GOOGLE_API_KEY:
        logger.error("Google API Key not configured. Cannot call Google API.")
        return None

    try:
        # Ensure genai library is configured with the key
        genai.configure(api_key=config.GOOGLE_API_KEY)

        # Create the specific model instance
        model = genai.GenerativeModel(config.LLM_MODEL_ID)

        # Configure generation parameters
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=config.LLM_MAX_NEW_TOKENS,
            temperature=config.LLM_TEMPERATURE
        )

        # Optional: Configure safety settings
        # safety_settings = { ... } # Refer to Google AI documentation

        # Make the API call
        response = model.generate_content(
            prompt,
            generation_config=generation_config#,
            # safety_settings=safety_settings
        )

        # Process response, checking for blocked content
        try:
            # Accessing response.text is the standard way, but can raise ValueError if blocked
            response_text = response.text
            logger.info("Google API request successful.")
            return response_text.strip()
        except ValueError:
            # Handle cases where response text is inaccessible (likely safety block)
            logger.warning("Google API response contained no text parts (likely blocked by safety filters).")
            if response.prompt_feedback:
                 logger.warning(f"Prompt Feedback: {response.prompt_feedback}")
            return None # Indicate failure or blocked content

    except Exception as e:
        # Catch other Google API errors (auth, quota, server issues, etc.)
        logger.error(f"Unexpected error during Google API call: {e}", exc_info=True)
        # Re-raise to allow tenacity retry
        raise e
# ---

# --- Main LLM Dispatcher Function ---
def get_llm_response(prompt: str) -> Union[str, None]:
    """
    Routes the prompt to the appropriate LLM API based on config.LLM_PROVIDER.

    Args:
        prompt: The input prompt string.

    Returns:
        The LLM's response string, or None if an error occurred.
    """
    provider = config.LLM_PROVIDER
    # logger.debug(f"Routing prompt to LLM Provider: {provider}") # Debug level

    if provider == "groq":
        return _call_groq_api(prompt)
    elif provider == "huggingface":
         if hasattr(config, 'HF_MODEL_ID') and hasattr(config, 'HF_API_URL'):
             return _call_huggingface_api(prompt)
         else:
             logger.error("Hugging Face selected, but config missing HF_MODEL_ID or HF_API_URL.")
             return None
    elif provider == "google":
        return _call_google_api(prompt)
    else:
        logger.error(f"Unsupported LLM provider configured: {provider}")
        return None