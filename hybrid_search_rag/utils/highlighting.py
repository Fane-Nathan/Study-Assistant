import nltk
from thefuzz import fuzz # Requires: pip install thefuzz python-Levenshtein
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Ensure sentence tokenizer is available (download if needed)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.warning("NLTK 'punkt' tokenizer not found for highlighting, attempting download.")
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        logger.error(f"Failed to download NLTK punkt for highlighting: {e}")
        # Highlighting might fail if download doesn't work

def highlight_sources_fuzzy(
    response_text: str,
    context_chunks: List[Dict[str, Any]],
    threshold: int = 85 # Similarity threshold (0-100), adjust based on testing
    ) -> str:
    """
    Adds simple Markdown highlighting to response sentences based on fuzzy matching
    with context sentences.

    Args:
        response_text: The full text response from the LLM.
        context_chunks: List of dictionaries representing the source chunks provided
                        to the LLM. Each dict MUST contain at least a 'text' key.
                        It's helpful if they also represent the original citation order (1, 2, 3...).
        threshold: The fuzz.token_set_ratio score needed to consider it a match.

    Returns:
        A string with matched sentences potentially wrapped in Markdown bold tags
        with citation numbers.
    """
    if not response_text or not context_chunks:
        logger.debug("Highlighting skipped: Empty response or context.")
        return response_text

    try:
        response_sentences = nltk.sent_tokenize(response_text)
        if not response_sentences:
            logger.warning("Could not tokenize response into sentences.")
            return response_text
    except Exception as e:
        logger.error(f"Failed to sentence tokenize LLM response: {e}")
        return response_text # Return original text if tokenization fails

    # Create a flat list of context sentences with their original chunk index (1-based)
    context_sentences_with_indices: List[Tuple[str, int]] = []
    for idx, chunk in enumerate(context_chunks):
        chunk_text = chunk.get('text', '')
        logger.debug(f"Highlighting loop - Chunk {idx} - Type: {type(chunk_text)}, Text: '{str(chunk_text)[:100]}...'")
        if not chunk_text or not isinstance(chunk_text, str):
             logger.warning(f"Skipping invalid context chunk at index {idx}")
             continue
        try:
            chunk_sentences = nltk.sent_tokenize(chunk_text)
            for sent in chunk_sentences:
                context_sentences_with_indices.append((sent, idx + 1)) # Store sentence and its 1-based chunk index
        except Exception as e:
             logger.warning(f"Failed to sentence tokenize context chunk {idx}: {e}")
             continue # Skip problematic chunks

    if not context_sentences_with_indices:
        logger.warning("No valid sentences extracted from context chunks for highlighting.")
        return response_text

    highlighted_response_parts = []
    matched_sources_overall = set() # Keep track of sources cited

    # Process each sentence in the LLM response
    for resp_sent in response_sentences:
        best_match_score = 0
        best_match_source_index = None

        # Find the *best* matching context sentence for this response sentence
        for ctx_sent, source_index in context_sentences_with_indices:
            # Use token_set_ratio: good for different word order, subset matches
            score = fuzz.token_set_ratio(resp_sent.lower(), ctx_sent.lower()) # Compare lowercased
            if score >= threshold and score > best_match_score:
                best_match_score = score
                best_match_source_index = source_index

        # If a good enough match was found, format it
        if best_match_source_index is not None:
            # Simple Markdown highlighting: Bold citation marker prepended
            highlighted_sentence = f"**[{best_match_source_index}]** {resp_sent}"
            highlighted_response_parts.append(highlighted_sentence)
            matched_sources_overall.add(best_match_source_index)
        else:
            # No good match found, add the original sentence without highlighting
            highlighted_response_parts.append(resp_sent)

    logger.info(f"Highlighting added for sources: {sorted(list(matched_sources_overall))}")
    # Join the processed sentences back together
    return " ".join(highlighted_response_parts)