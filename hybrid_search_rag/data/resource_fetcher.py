# hybrid_search_rag/data/resource_fetcher.py
"""Fetches paper metadata from arXiv and web content."""

import re
import arxiv
import fitz
import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Any, Optional, Tuple # Correct imports for Python 3.9
# import concurrent.futures
import time
import io
from urllib.parse import urljoin, urlparse
from collections import deque

logger = logging.getLogger(__name__)

FETCH_TIMEOUT = 60
HEAD_TIMEOUT = 10
# --- Crawling Settings (NEW) ---
MAX_PAGES_TO_CRAWL = 50 # Limit the total number of pages fetched per run
CRAWL_DELAY_SECONDS = 1 # Politeness delay between requests
ALLOWED_DOMAINS = [] # Optional: If empty, allows same domain as start URL. If populated, only these domains.


def _clean_text(text: str) -> str:
    """
    Performs basic cleaning of extracted text from web pages or PDFs.

    - Normalizes whitespace (spaces, tabs).
    - Removes leading/trailing whitespace from lines.
    - Reduces multiple consecutive newlines to a maximum of two (paragraph breaks).
    """
    if not text:
        return ""

    try:
        # 1. Replace tabs and multiple spaces with a single space
        text = re.sub(r'[ \t]+', ' ', text)

        # 2. Split text into lines
        lines = text.splitlines()

        # 3. Strip leading/trailing whitespace from each line and keep non-empty lines
        cleaned_lines = [line.strip() for line in lines]
        cleaned_lines = [line for line in cleaned_lines if line] # Filter out empty lines

        if not cleaned_lines:
            return ""

        # 4. Join lines back with single newlines
        text = "\n".join(cleaned_lines)

        # 5. Reduce sequences of 3 or more newlines down to exactly 2 (to preserve paragraph breaks)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 6. Final strip to remove any leading/trailing whitespace from the whole block
        return text.strip()

    except Exception as e:
        logger.error(f"Error during text cleaning: {e}")
        # Return the original text or empty string in case of unexpected error
        return text or ""

# --- This is the function that handles both PDF and HTML ---
def _fetch_web_content(url: str) -> Tuple[Optional[str], str, List[str]]:
    """
    Fetches content from a URL, attempting to handle HTML or PDF.
    ALWAYS returns (content, final_url, discovered_links_list).
    Links are only discovered for successful HTML fetches.
    """
    logger.info(f"Fetching web content from: {url}")
    headers = {'User-Agent': 'Mozilla/5.0'}
    discovered_links: List[str] = []
    final_url = url # Store the potentially redirected URL

    # print(f"--- DEBUG: Starting _fetch_web_content for {url} ---") # Keep debug prints for now

    try:
        # --- Check content type (HEAD request) ---
        content_type = ''
        try:
            # print("DEBUG: Checking HEAD request for Content-Type")
            head_response = requests.head(url, headers=headers, timeout=HEAD_TIMEOUT, allow_redirects=True)
            head_response.raise_for_status()
            content_type = head_response.headers.get('Content-Type', '').lower()
            final_url = head_response.url # Update final_url after redirects
            # print(f"DEBUG: HEAD Content-Type: {content_type} for final URL: {final_url}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"HEAD request failed for {url}: {e}. Cannot determine type reliably.")
            # Return None content, the original url, and empty links
            return None, url, []

        is_pdf = 'application/pdf' in content_type

        # --- Process based on type ---
        # print(f"DEBUG: Processing as {'PDF' if is_pdf else 'HTML'}")
        response = requests.get(final_url, headers=headers, timeout=FETCH_TIMEOUT)
        response.raise_for_status()
        final_url = response.url # Update again in case GET redirected differently

        if is_pdf:
            pdf_text = ""
            try:
                with fitz.open(stream=response.content, filetype="pdf") as doc:
                    logger.info(f"Extracting text from {doc.page_count} pages in PDF: {final_url}")
                    for i, page in enumerate(doc):
                        try:
                            page_text = page.get_text("text", sort=True) # Added sort=True for potentially better reading order
                            if page_text: pdf_text += page_text + "\n\n" # Use \n\n as page separator
                        except Exception as page_err:
                            logger.warning(f"Could not extract text from page {i+1} of {final_url}: {page_err}")
                
                cleaned_text = _clean_text(pdf_text) if pdf_text else None
                if cleaned_text:
                    logger.info(f"Successfully extracted and cleaned text from PDF: {final_url}")
                    return cleaned_text, final_url, [] # PDF -> No discovered links
                else:
                    logger.warning(f"No text could be extracted from PDF: {final_url}")
                    return None, final_url, []

            except Exception as e:
                logger.error(f"Failed to process PDF {final_url}: {e}", exc_info=True)
                return None, final_url, []

        else: # --- Assume HTML ---
            try:
                response.encoding = response.apparent_encoding # Guess encoding
                soup = BeautifulSoup(response.text, 'html.parser')
                main_content_area = soup.find('article') or soup.find('main') or soup.body

                cleaned_text: Optional[str] = None
                if main_content_area:
                    raw_text = main_content_area.get_text(separator='\n', strip=True)
                    cleaned_text = _clean_text(raw_text)
                    logger.info(f"Successfully fetched and cleaned HTML content from: {final_url}")

                    # --- Discover Links (only if content was found) ---
                    base_url = final_url # Use the final URL after redirects as base
                    for link in main_content_area.find_all('a', href=True):
                        href = link['href'].strip()
                        if href and not href.startswith('#') and not href.startswith(('mailto:', 'javascript:')):
                            absolute_url = urljoin(base_url, href)
                            discovered_links.append(absolute_url)
                    logger.debug(f"Discovered {len(discovered_links)} potential links in {final_url}")

                else:
                    logger.warning(f"Could not find main content tags in HTML: {final_url}")

                # Return content (or None) and any discovered links
                return cleaned_text, final_url, discovered_links

            except Exception as e:
                logger.error(f"Failed to parse HTML {final_url}: {e}", exc_info=True)
                return None, final_url, [] # Return empty links on error

    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for URL {url} (final URL attempted: {final_url}): {e}")
        return None, final_url, []
    except Exception as e:
        logger.error(f"Unexpected error processing URL {url}: {e}", exc_info=True)
        return None, url, [] # Use original url if final_url wasn't set
    # finally:
        #  print(f"--- DEBUG: Finishing _fetch_web_content for {url} ---")


# --- Helper function for domain checking (NEW) ---
def get_domain(url: str) -> Optional[str]:
    """Extracts the domain name (e.g., 'google.com') from a URL."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return None

# --- Modified fetch_web_articles to implement crawling (RENAMED and CHANGED) ---
def crawl_and_fetch_web_articles(start_urls: List[str]) -> List[Dict[str, Any]]:
    """
    Crawls web URLs starting from start_urls, fetching content sequentially.
    Respects MAX_PAGES_TO_CRAWL, CRAWL_DELAY_SECONDS, and basic domain filtering.
    Implements early stopping once the visited URL limit is reached.
    """
    if not start_urls:
        logger.info("No starting web URLs provided for crawling.")
        return []

    results_list: List[Dict[str, Any]] = []
    queue = deque(start_urls)
    visited = set(start_urls)
    start_domains = {get_domain(url) for url in start_urls if get_domain(url)} # Domains of initial URLs

    # --- MODIFICATION: Add early stopping flag ---
    stop_adding_links_flag = False
    # ---

    # Define the limit for the visited set size
    visited_set_limit = MAX_PAGES_TO_CRAWL * 2 # Safety limit for exploration

    logger.info(f"Starting crawl from {len(start_urls)} URLs. Max pages: {MAX_PAGES_TO_CRAWL}. Delay: {CRAWL_DELAY_SECONDS}s. Visited limit: {visited_set_limit}.")

    while queue and len(results_list) < MAX_PAGES_TO_CRAWL:

        # --- MODIFICATION: Check early stopping flag at the start of the loop ---
        if stop_adding_links_flag:
            logger.info("Early stopping: Visited set limit was reached previously. Halting crawl.")
            break # Exit the main while loop
        # ---

        current_url = queue.popleft()
        logger.info(f"Processing URL from queue ({len(queue)} left): {current_url}")

        # --- Politeness Delay ---
        time.sleep(CRAWL_DELAY_SECONDS)

        # --- Fetch content and discover links ---
        try:
            content, actual_url, links = _fetch_web_content(current_url)

            # Ensure actual_url (after redirects) is also marked visited
            if actual_url != current_url and actual_url not in visited:
                # Add to visited only if below limit, otherwise we might exceed it here
                if len(visited) < visited_set_limit:
                     visited.add(actual_url)
                     logger.debug(f"Added redirected URL to visited: {actual_url}")
                # If already at limit, don't add, but proceed to process content if fetched
                # The main check below will handle the stop_adding_links_flag

            # --- Process fetched content ---
            if content:
                # Add to results only if we haven't reached the max pages limit
                if len(results_list) < MAX_PAGES_TO_CRAWL:
                    title = actual_url # Use final URL as title for simplicity now
                    source_type = "pdf" if actual_url.lower().endswith('.pdf') else "web" # Check actual_url
                    result_data = {
                        "source": source_type,
                        "url": actual_url, # Store the final URL
                        "title": title,
                        "content": f"{title}. {content}",
                        "authors": [],
                        "published": None,
                    }
                    results_list.append(result_data)
                    logger.info(f"Successfully processed content from URL: {actual_url} ({len(results_list)}/{MAX_PAGES_TO_CRAWL})")
                else:
                     # If we just hit the results limit, stop the crawl immediately
                     logger.info(f"Reached MAX_PAGES_TO_CRAWL ({MAX_PAGES_TO_CRAWL}). Halting crawl.")
                     break # Exit the main while loop


                # --- Process discovered links (only if content was successfully processed) ---
                current_domain = get_domain(actual_url)
                for link in links:
                    # Basic Filtering (can be significantly expanded)
                    # 1. Check if already visited
                    if link in visited:
                        continue

                    # 2. Basic scheme check
                    if not link.startswith(('http://', 'https://')):
                         continue

                    # 3. File extension check (basic)
                    parsed_link_path = urlparse(link).path.lower()
                    if any(parsed_link_path.endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.zip', '.css', '.js', '.xml', '.json', '.txt', '.gif']):
                         continue # Skip common non-HTML files for now (can adjust if PDF crawling needed)

                    # 4. Domain check: Only crawl within the same domain(s) as the start URLs OR specific allowed domains
                    link_domain = get_domain(link)
                    if not link_domain: # Cannot parse domain
                         continue

                    if ALLOWED_DOMAINS: # If a whitelist is specified
                         if link_domain not in ALLOWED_DOMAINS:
                              continue # Skip if not in explicit whitelist
                    elif start_domains: # Otherwise, if initial domains were found
                         if link_domain not in start_domains:
                              # Optional: could allow subdomains here if needed
                              continue # Skip if not in the same set of domains we started with
                    # If ALLOWED_DOMAINS is empty AND start_domains is empty (shouldn't happen with valid start URLs), allow all? Or block all? Let's block.
                    elif not start_domains:
                         continue

                    # If passes filters: Add to queue and visited set *IF* limit not reached
                    # --- MODIFICATION: Check limit and set flag ---
                    if len(visited) < visited_set_limit:
                         visited.add(link)
                         queue.append(link)
                         logger.debug(f"Queued valid link: {link}")
                    else:
                         # Check flag to log warning only once if desired (optional)
                         # if not stop_adding_links_flag:
                         #      logger.warning("Visited set limit reached, not adding more links.")
                         logger.warning("Visited set limit reached, not adding more links.") # Log every time for now
                         stop_adding_links_flag = True # Set the flag
                         break # Stop processing links *for this page*
                    # ---

            else:
                logger.warning(f"Processing URL {actual_url} resulted in no extractable content.")

        except Exception as e:
            logger.error(f"Crawling/processing {current_url} generated an exception: {e}", exc_info=True)

    logger.info(f"Crawling finished. Successfully processed content from {len(results_list)} pages.")
    return results_list

# --- arXiv Fetcher ---
def fetch_arxiv_papers(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Fetches paper metadata from arXiv based on a query."""
    logger.info(f"Fetching up to {max_results} papers from arXiv for query: '{query}'")
    if max_results <= 0:
        return []

    client = arxiv.Client(page_size = min(max_results, 1000), delay_seconds = 2, num_retries = 3)
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)

    results_list: List[Dict[str, Any]] = []
    fetched_count = 0
    try:
        for result in client.results(search):
            if fetched_count >= max_results:
                break

            clean_summary = (result.summary or "").replace('\n', ' ').strip()
            title = (result.title or "Untitled").strip()
            # Prepend title to content for consistency
            content = f"{title}. {clean_summary}"

            paper_data = {
                "source": "arxiv",
                "entry_id": result.entry_id,
                "title": title,
                "content": content,
                "authors": [str(author) for author in result.authors],
                "published": result.published.isoformat() if result.published else None,
                "url": result.pdf_url or result.entry_id
            }
            results_list.append(paper_data)
            fetched_count += 1
        logger.info(f"Successfully fetched {len(results_list)} papers from arXiv.")
    except Exception as e:
        logger.error(f"An error occurred while fetching from arXiv: {e}", exc_info=True)
    finally:
        return results_list