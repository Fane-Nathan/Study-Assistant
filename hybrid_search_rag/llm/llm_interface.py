# hybrid_search_rag/llm/llm_interface.py
"""Interface for interacting with different Large Language Model APIs."""

import logging
import requests
from groq import Groq, RateLimitError, APIError
from tenacity import retry, stop_after_attempt, wait_exponential # For potential retries
from .. import config # Use relative import assuming standard package structure
from typing import Union

# --- ADDED: Import Google Generative AI library ---
import google.generativeai as genai
# ---

logger = logging.getLogger(__name__)

# --- Groq API Call Function (Existing - No changes needed here) ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def _call_groq_api(prompt: str) -> Union[str, None]:
    """Makes an API call to the Groq API using configured settings."""
    logger.info(f"Sending request to Groq API (Model: {config.LLM_MODEL_ID})...")

    if not config.GROQ_API_KEY:
        logger.error("Groq API Key (GROQ_API_KEY) is not configured.")
        return None

    try:
        client = Groq(api_key=config.GROQ_API_KEY, timeout=config.LLM_API_TIMEOUT)

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=config.LLM_MODEL_ID,
            max_tokens=config.LLM_MAX_NEW_TOKENS,
            temperature=config.LLM_TEMPERATURE,
            stream=False,
        )

        response_text = chat_completion.choices[0].message.content
        logger.info(f"Groq API request successful. Response length: {len(response_text)}")
        return response_text.strip()

    except RateLimitError as e:
        logger.error(f"Groq API Rate Limit Error: {e}. Check your usage limits on console.groq.com.")
        raise e
    except APIError as e:
        logger.error(f"Groq API Error (Status {e.status_code}): {e.message}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during Groq API call: {e}", exc_info=True)
        return None

# --- Hugging Face API Call Function (Existing - No changes needed here) ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def _call_huggingface_api(prompt: str) -> Union[str, None]:
    """Makes an API call to the Hugging Face Inference API."""
    logger.info(f"Sending request to HF Inference API ({config.HF_MODEL_ID})...") # Use HF_MODEL_ID here

    # Make sure HF_MODEL_ID and HF_API_URL are defined in config if using this
    if not hasattr(config, 'HF_MODEL_ID') or not hasattr(config, 'HF_API_URL'):
         logger.error("Hugging Face provider selected, but HF_MODEL_ID or HF_API_URL not defined in config.")
         return None
    if not config.HF_API_TOKEN or not config.HF_API_URL: # Need URL too
        logger.error("Hugging Face API Token or URL is not configured.")
        return None

    headers = {"Authorization": f"Bearer {config.HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": config.LLM_MAX_NEW_TOKENS,
            "temperature": config.LLM_TEMPERATURE if config.LLM_TEMPERATURE > 0 else None, # Some models fail if temp is 0
            "return_full_text": False,
        },
        "options": {"wait_for_model": True}
    }
    # Remove None temperature if it was 0 or less
    if payload["parameters"]["temperature"] is None:
        del payload["parameters"]["temperature"]

    try:
        response = requests.post(
            config.HF_API_URL,
            headers=headers,
            json=payload,
            timeout=config.LLM_API_TIMEOUT
        )
        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and result:
            generated_text = result[0].get("generated_text")
        elif isinstance(result, dict):
             generated_text = result.get("generated_text")
        else:
            generated_text = None

        if generated_text:
            logger.info(f"HF API request successful. Response length: {len(generated_text)}")
            return generated_text.strip()
        else:
            logger.error(f"HF API Error: Unexpected response format or empty result: {result}")
            return None

    except requests.exceptions.HTTPError as e:
        error_details = e.response.text
        logger.error(f"HF API request failed: Status {e.response.status_code}. Details: {error_details}")
        if e.response.status_code == 429: raise e
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"HF API request failed due to network issue or timeout: {e}", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred during HF API call: {e}", exc_info=True)
        return None

# --- ADDED: Google Generative AI (Gemini) API Call Function ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def _call_google_api(prompt: str) -> Union[str, None]:
    """Makes an API call to the Google Generative AI API using configured settings."""
    logger.info(f"Sending request to Google API (Model: {config.LLM_MODEL_ID})...")

    if not config.GOOGLE_API_KEY:
        logger.error("Google API Key (GOOGLE_API_KEY) is not configured.")
        return None

    try:
        # Configure the library with the API key
        genai.configure(api_key=config.GOOGLE_API_KEY)

        # Create the model instance
        model = genai.GenerativeModel(config.LLM_MODEL_ID)

        # Set up generation configuration from config.py
        # Note: Parameter names might differ slightly (e.g., max_new_tokens -> max_output_tokens)
        generation_config = genai.types.GenerationConfig(
            # candidate_count=1, # Default is 1
            # stop_sequences=None, # Optional stop sequences
            max_output_tokens=config.LLM_MAX_NEW_TOKENS,
            temperature=config.LLM_TEMPERATURE
            # top_p=None, # Optional Top-P sampling
            # top_k=None  # Optional Top-K sampling
        )

        # Safety settings (optional, defaults are usually balanced)
        # You can customize these based on Google's documentation if needed
        # safety_settings = {
        #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        #     # ... other categories
        # }

        # Make the API call
        response = model.generate_content(
            prompt,
            generation_config=generation_config#,
            # safety_settings=safety_settings # Uncomment if using custom safety settings
        )

        # Extract the text response - important to check it exists
        # The API might block a response due to safety, returning no .text
        if response.parts:
             response_text = response.text # .text joins parts automatically
             logger.info(f"Google API request successful. Response length: {len(response_text)}")
             return response_text.strip()
        else:
             # Log potentially blocked content or other issues
             logger.warning(f"Google API response received, but contained no usable text parts. Check prompt feedback: {response.prompt_feedback}")
             return None # Indicate failure or lack of content

    except Exception as e:
        # Catching general exceptions for now. Google's library might raise specific errors
        # like google.api_core.exceptions.PermissionDenied, ResourceExhausted, etc.
        # which could be handled more granularly later.
        logger.error(f"An unexpected error occurred during Google API call: {e}", exc_info=True)
        # Consider if specific errors should allow retries (e.g., ResourceExhausted might be temporary)
        # For now, return None on any exception during the call.
        return None
# ---


# --- Main Function to Get LLM Response ---
def get_llm_response(prompt: str) -> Union[str, None]:
    """
    Gets a response from the configured LLM provider based on config.LLM_PROVIDER.

    Args:
        prompt: The input prompt string for the LLM.

    Returns:
        The LLM's response string, or None if an error occurred.
    """
    logger.debug(f"LLM Provider selected: {config.LLM_PROVIDER}")

    if config.LLM_PROVIDER == "groq":
        return _call_groq_api(prompt)
    elif config.LLM_PROVIDER == "huggingface":
         # Make sure HF_MODEL_ID and HF_API_URL are defined in config if using this
         if hasattr(config, 'HF_MODEL_ID') and hasattr(config, 'HF_API_URL'):
             return _call_huggingface_api(prompt)
         else:
             logger.error("Hugging Face provider selected, but HF_MODEL_ID or HF_API_URL not defined in config.")
             return None
    elif config.LLM_PROVIDER == "google": # <-- MODIFIED: Added google provider option
        return _call_google_api(prompt)
    else:
        # This was the block causing the original error
        logger.error(f"Unsupported LLM provider configured: {config.LLM_PROVIDER}")
        return None