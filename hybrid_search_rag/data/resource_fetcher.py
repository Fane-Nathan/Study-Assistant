# hybrid_search_rag/data/resource_fetcher.py
"""Fetches data from arXiv and web sources (HTML/PDF) asynchronously."""

import asyncio
import io
import logging
import os
import random
import re
import time # Import time for rate limiting
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Set # Added Set
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import aiohttp
import arxiv
import fitz
from bs4 import BeautifulSoup
from cachetools import LRUCache
from pybloom import BloomFilter
from readability import Document as ReadabilityDocument
from prometheus_client import Counter, Histogram
from trafilatura import extract
from trafilatura.settings import use_config
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# --- Load configuration ---
# Assuming config.py exists in the parent directory or values are defined below
try:
    from .. import config
    FETCH_TIMEOUT = config.FETCH_TIMEOUT
    HEAD_TIMEOUT = config.HEAD_TIMEOUT
    MAX_PAGES_TO_CRAWL = config.MAX_PAGES_TO_CRAWL
    CRAWL_DELAY_SECONDS = config.CRAWL_DELAY_SECONDS
    ALLOWED_DOMAINS = config.ALLOWED_DOMAINS # Optional: List of allowed domains
except ImportError:
    logger.warning("config.py not found or variables missing. Using default values.")
    FETCH_TIMEOUT = 30
    HEAD_TIMEOUT = 15
    MAX_PAGES_TO_CRAWL = 100
    CRAWL_DELAY_SECONDS = 1.0 # Default 1 second delay between requests
    ALLOWED_DOMAINS = None # Set to a list of strings like ["example.com", "anothersite.org"] to restrict crawling


# --- Request Settings ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
    # Add more diverse user agents if needed
]

def get_user_agent() -> str:
    """Selects a random User-Agent string."""
    if not USER_AGENTS:
        return "Mozilla/5.0 (compatible; Python Fetcher)" # Fallback
    return random.choice(USER_AGENTS)

# --- Crawling Settings ---
LAST_REQUEST_TIME: float = 0.0 # For rate limiting

# --- Constants for Content Types ---
CONTENT_TYPE_HTML = "html"
CONTENT_TYPE_PDF = "pdf"
CONTENT_TYPE_OTHER = "other"
CONTENT_TYPE_FAILED = "failed"
CONTENT_TYPE_ARXIV = "arxiv"

# --- Metrics ---
# (Keep your Prometheus metric definitions)
REQUESTS_TOTAL = Counter(
    "fetcher_requests_total", "Total number of requests made by the fetcher", ["status"]
)
REQUEST_DURATION = Histogram(
    "fetcher_request_duration_seconds", "Duration of requests made by the fetcher"
)
PAGES_CRAWLED_TOTAL = Counter(
    "fetcher_pages_crawled_total", "Total number of pages crawled by the fetcher"
)
PAGES_FAILED_TOTAL = Counter(
    "fetcher_pages_failed_total", "Total number of pages failed by the fetcher"
)

# --- Caching ---
CACHE_MAX_SIZE = 200 # Increased cache size
TRAFILATURA_CONFIG = use_config()
TRAFILATURA_CONFIG.set("DEFAULT", "EXTRACTION_TIMEOUT", "60") # Set timeout for Trafilatura
# Cache stores: (final_url, content_type, text_content, raw_content_bytes_or_None)
resource_cache: LRUCache = LRUCache(maxsize=CACHE_MAX_SIZE)

# --- Text Cleaning ---
def _clean_text(text: Optional[str]) -> Optional[str]:
    """Cleans extracted text by normalizing whitespace and reducing excessive newlines."""
    if not text: return None
    try:
        # Replace multiple spaces/tabs with a single space
        text = re.sub(r'[ \t]+', ' ', text)
        lines = text.splitlines()
        # Strip whitespace from each line and remove empty lines
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        if not cleaned_lines: return None # Return None if cleaning results in empty text
        # Join lines back, ensuring single newlines between originally separate lines
        text = "\n".join(cleaned_lines)
        # Reduce 3 or more consecutive newlines (paragraph breaks) to exactly 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Final strip to remove leading/trailing whitespace
        cleaned_text = text.strip()
        return cleaned_text if cleaned_text else None # Return None if final result is empty
    except Exception as e:
        logger.error(f"Text cleaning failed: {e}")
        return text # Return original text on error

# --- Retry Decorator ---
retry_decorator = retry(
    stop=stop_after_attempt(3), # Reduced attempts slightly
    wait=wait_exponential(multiplier=1, min=2, max=10), # Adjusted wait times
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    reraise=True # Reraise the exception after retries are exhausted
)

# --- Core Fetcher ---
@retry_decorator
@REQUEST_DURATION.time()
async def _fetch_and_parse_content(url: str, session: aiohttp.ClientSession, proxy: Optional[str] = None) -> Tuple[str, str, Optional[str], Optional[bytes]]:
    """
    Fetches URL, determines type, extracts text, and returns parsed content asynchronously.

    Returns:
        Tuple[str, str, Optional[str], Optional[bytes]]:
        (final_url, content_type_constant, text_content, raw_content_bytes_or_None)
        raw_content_bytes_or_None is the raw byte content for HTML, None otherwise.
    """
    logger.info(f"Fetching: {url}")
    # Check cache first
    if url in resource_cache:
        logger.info(f"Cache hit for: {url}")
        cached_data = resource_cache[url]
        # Ensure cache returns the expected 4 values
        if isinstance(cached_data, tuple) and len(cached_data) == 4:
             return cached_data # type: ignore
        else:
             logger.warning(f"Invalid cache format for {url}, re-fetching.")
             del resource_cache[url] # Remove invalid entry

    user_agent = get_user_agent()
    headers: Dict[str, str] = {'User-Agent': user_agent}
    REQUESTS_TOTAL.labels(status="started").inc()

    final_url: str = url
    raw_content: Optional[bytes] = None

    try:
        # === HEAD Request ===
        content_type_header: str = ""
        try:
            logger.debug(f"Sending HEAD request to {url} with proxy {proxy}")
            async with session.head(url, headers=headers, timeout=HEAD_TIMEOUT, allow_redirects=True, proxy=proxy) as head_response:
                head_response.raise_for_status() # Check for 4xx/5xx errors
                content_type_header = head_response.headers.get("Content-Type", "").lower()
                final_url = str(head_response.url) # Update final URL after redirects
                logger.debug(f"HEAD success for {url}. Final URL: {final_url}, Content-Type: {content_type_header}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"HEAD request failed for {url}: {e}. Aborting fetch.")
            REQUESTS_TOTAL.labels(status="failed").inc()
            # Return FAILED, original URL as final_url might not be updated
            return url, CONTENT_TYPE_FAILED, None, None
        except Exception as e:
            logger.error(f"Unexpected error during HEAD request for {url}: {e}", exc_info=True)
            REQUESTS_TOTAL.labels(status="failed").inc()
            return url, CONTENT_TYPE_FAILED, None, None


        is_pdf = 'application/pdf' in content_type_header
        is_html = 'text/html' in content_type_header

        # === GET Request (only if HTML or PDF) ===
        if is_html or is_pdf:
            try:
                logger.debug(f"Sending GET request to {final_url} with proxy {proxy}")
                async with session.get(final_url, headers=headers, timeout=FETCH_TIMEOUT, allow_redirects=True, proxy=proxy) as response:
                    response.raise_for_status() # Check for 4xx/5xx errors
                    final_url = str(response.url) # Update final URL again after GET redirects
                    raw_content = await response.read() # Read content bytes
                    logger.debug(f"GET success for {final_url}. Read {len(raw_content)} bytes.")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"GET request failed for {final_url}: {e}")
                REQUESTS_TOTAL.labels(status="failed").inc()
                return final_url, CONTENT_TYPE_FAILED, None, None # Return FAILED
            except Exception as e:
                 logger.error(f"Unexpected error during GET request for {final_url}: {e}", exc_info=True)
                 REQUESTS_TOTAL.labels(status="failed").inc()
                 return final_url, CONTENT_TYPE_FAILED, None, None

        else:
             # Handle non-HTML/PDF types determined by HEAD request
             logger.info(f"Skipping content fetch for non-PDF/HTML type '{content_type_header}' at {final_url}")
             REQUESTS_TOTAL.labels(status="skipped").inc()
             resource_cache[url] = (final_url, CONTENT_TYPE_OTHER, None, None)
             return final_url, CONTENT_TYPE_OTHER, None, None


        # === Process Based on Type (using raw_content) ===
        if is_pdf and raw_content:
            pdf_text: str = ""
            try:
                # Use raw_content bytes directly with fitz
                with fitz.open(stream=io.BytesIO(raw_content), filetype="pdf") as doc: # Use BytesIO stream
                    # Limit pages? doc.page_count
                    for i, page in enumerate(doc):
                        # if i >= MAX_PDF_PAGES: break # Optional page limit
                        pdf_text += page.get_text("text", sort=True) + "\n\n"

                cleaned_text: Optional[str] = _clean_text(pdf_text)
                if cleaned_text:
                    REQUESTS_TOTAL.labels(status="success").inc()
                    logger.info(f"Successfully processed PDF: {final_url}")
                    # Cache includes None for raw_content for PDFs after processing
                    resource_cache[url] = (final_url, CONTENT_TYPE_PDF, cleaned_text, None)
                    return final_url, CONTENT_TYPE_PDF, cleaned_text, None
                else:
                    logger.warning(f"No text extracted from PDF: {final_url}")
                    PAGES_FAILED_TOTAL.inc()
                    return final_url, CONTENT_TYPE_PDF, None, None # Return PDF type but no content
            except Exception as e:
                logger.error(f"PDF processing failed for {final_url}: {e}", exc_info=True)
                PAGES_FAILED_TOTAL.inc()
                return final_url, CONTENT_TYPE_PDF, None, None # Return PDF type but no content

        elif is_html and raw_content:
            cleaned_text: Optional[str] = None
            try:
                # 1. Attempt extraction with Readability & Trafilatura
                try:
                    # Use Readability first to check if there's meaningful content
                    readable_doc = ReadabilityDocument(raw_content, url=final_url)
                    # readable_doc.summary() often gives good results, check if it's non-empty
                    if readable_doc.summary():
                        # If Readability finds something, use Trafilatura on raw bytes for main content
                        extracted_text: Optional[str] = extract(raw_content,
                                                               config=TRAFILATURA_CONFIG,
                                                               favor_recall=True,
                                                               include_comments=False, # Don't include comments
                                                               include_tables=True, # Include table text
                                                               url=final_url) # Provide URL context
                        cleaned_text = _clean_text(extracted_text)
                        if cleaned_text:
                             logger.debug(f"Extracted text using Trafilatura for {final_url}")
                        else:
                             logger.debug(f"Trafilatura extracted no text, falling back for {final_url}")

                except Exception as extraction_err:
                     logger.warning(f"Readability/Trafilatura failed for {final_url}: {extraction_err}. Falling back to basic parsing.")
                     cleaned_text = None # Ensure fallback happens

                # 2. Fallback to basic BeautifulSoup if advanced extraction failed or yielded no text
                if not cleaned_text:
                    logger.info(f"Falling back to basic BS4 text extraction for {final_url}")
                    soup_fallback = BeautifulSoup(raw_content, 'html.parser')
                    # Try common main content tags, default to body
                    main_content_area = soup_fallback.find('article') or soup_fallback.find('main') or soup_fallback.body
                    if main_content_area:
                        raw_text_fallback: str = main_content_area.get_text(separator='\n', strip=True)
                        cleaned_text = _clean_text(raw_text_fallback)
                        if cleaned_text:
                             logger.debug(f"Extracted text using BS4 fallback for {final_url}")
                    if not cleaned_text: # Check again if fallback failed
                         logger.warning(f"Could not extract text using BS4 fallback for {final_url}")

                # 3. Return result
                if cleaned_text:
                    REQUESTS_TOTAL.labels(status="success").inc()
                    logger.info(f"Successfully processed HTML: {final_url}")
                    # Cache extracted text, pass raw_content for link extraction later
                    resource_cache[url] = (final_url, CONTENT_TYPE_HTML, cleaned_text, raw_content)
                    return final_url, CONTENT_TYPE_HTML, cleaned_text, raw_content
                else:
                    # If even fallback failed, count as failure but return raw content
                    logger.warning(f"No text content could be extracted from HTML: {final_url}")
                    PAGES_FAILED_TOTAL.inc()
                    resource_cache[url] = (final_url, CONTENT_TYPE_HTML, None, raw_content) # Cache failure state
                    return final_url, CONTENT_TYPE_HTML, None, raw_content # Return HTML type, no text, but raw content

            except Exception as e:
                logger.error(f"HTML parsing failed unexpectedly for {final_url}: {e}", exc_info=True)
                PAGES_FAILED_TOTAL.inc()
                # Return HTML type, failed state, but keep raw content if available
                return final_url, CONTENT_TYPE_HTML, None, raw_content

        else: # Should not happen if HEAD/GET logic is correct, but as safeguard
             logger.warning(f"Unexpected state: Reached end of fetch function for {final_url} with type '{content_type_header}' and raw_content presence: {raw_content is not None}")
             REQUESTS_TOTAL.labels(status="skipped").inc()
             return final_url, CONTENT_TYPE_OTHER, None, None

    # Outer exception handlers catch errors not caught by retry or specific handlers
    except aiohttp.ClientResponseError as http_err:
        logger.error(f"HTTP Error {http_err.status} for URL {url} (final URL tried: {final_url}): {http_err}")
        REQUESTS_TOTAL.labels(status="failed").inc()
        return final_url if 'final_url' in locals() else url, CONTENT_TYPE_FAILED, None, None
    except (aiohttp.ClientError, asyncio.TimeoutError) as e: # Catching ClientError covers more network issues
        logger.error(f"Network/Client error for URL {url} (final URL tried: {final_url}): {e}")
        REQUESTS_TOTAL.labels(status="failed").inc()
        return final_url if 'final_url' in locals() else url, CONTENT_TYPE_FAILED, None, None
    except Exception as e:
        logger.error(f"Unexpected error processing URL {url}: {e}", exc_info=True)
        REQUESTS_TOTAL.labels(status="failed").inc()
        # Use url as final_url might not be set
        return url, CONTENT_TYPE_FAILED, None, None

# --- Helper for Domain Checking ---
def get_domain(url: str) -> Optional[str]:
    """Extracts the 'netloc' (e.g., 'google.com') from a URL."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        logger.warning(f"Could not parse domain from URL: {url}")
        return None

# --- Robots.txt Handling ---
# NOTE: This is a SYNCHRONOUS check. For high-performance async crawling,
# consider using an async robots.txt library (like aiorobots) or running
# this check in a thread pool executor (asyncio.to_thread).
def _is_url_allowed_by_robots(base_url: str, check_url: str, user_agent: str = '*') -> bool:
    """Checks if a URL is allowed to be crawled based on robots.txt rules."""
    try:
        parsed_base = urlparse(base_url)
        robots_url = f"{parsed_base.scheme}://{parsed_base.netloc}/robots.txt"

        rp = RobotFileParser()
        rp.set_url(robots_url)
        # This read() call is blocking
        rp.read()
        # Use the specific user agent for checking if possible, else wildcard
        return rp.can_fetch(user_agent, check_url)
    except Exception as e:
        logger.warning(f"Error checking robots.txt for {base_url} ({robots_url}): {e}. Assuming allowed.")
        return True # Default to allowing if robots.txt check fails

# --- Web Crawler ---

# Define _process_url function within the scope where needed variables are accessible
# or pass them explicitly as arguments. Here it's defined before the main crawl function.
async def _process_url(
    current_url: str,
    session: aiohttp.ClientSession,
    results_list: List[Dict[str, Any]],
    visited: BloomFilter,
    queue: deque[str],
    start_domains: Set[Optional[str]],
    visited_set_limit: int,
    stop_event: asyncio.Event,
    semaphore: asyncio.Semaphore,
    proxy: Optional[str] = None
):
    """Process a single URL asynchronously: fetch, parse, extract links."""
    global LAST_REQUEST_TIME # Needed for rate limiting

    # Use semaphore to limit concurrent fetches/processing steps
    async with semaphore:
        # Check stop event *before* doing anything else for this URL
        if stop_event.is_set():
            logger.debug(f"Stopping processing for {current_url} as stop event is set.")
            return

        # Add to visited *before* fetching to avoid duplicate fetches in flight
        # Note on Bloom Filter: False positives are possible but low.
        # Adding here prevents *starting* work on a URL likely seen.
        if current_url in visited:
             logger.debug(f"Skipping already visited (in bloom filter): {current_url}")
             return
        visited.add(current_url) # Add original URL

        # --- Rate Limiting ---
        loop = asyncio.get_event_loop()
        now = loop.time()
        time_since_last = now - LAST_REQUEST_TIME
        if time_since_last < CRAWL_DELAY_SECONDS:
            sleep_duration = CRAWL_DELAY_SECONDS - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_duration:.2f}s")
            await asyncio.sleep(sleep_duration)
        LAST_REQUEST_TIME = loop.time() # Update last request time *after* potential sleep
        # --- End Rate Limiting ---
        
        soup: Optional[BeautifulSoup] = None

        try:
            # Fetch content, potentially receiving raw_content for HTML
            final_url, content_type, text_content, raw_content = await _fetch_and_parse_content(current_url, session, proxy=proxy)

            # Add final URL to visited as well, in case of redirects
            if final_url != current_url:
                 # Check again if the *final* URL was already visited (less likely with bloom filter, but possible)
                 if final_url in visited:
                     logger.debug(f"Skipping {current_url} because final URL {final_url} already visited.")
                     return # Avoid processing the same content twice via redirect
                 visited.add(final_url)

            # --- Process Content ---
            if content_type != CONTENT_TYPE_FAILED:
                # Check if we should add the result (stop event not set AND below limit)
                should_add_result = False
                if not stop_event.is_set():
                     # Check length before potentially adding
                     if len(results_list) < MAX_PAGES_TO_CRAWL:
                          should_add_result = True
                     else:
                          # We hit the limit *before* processing this page's content
                          # Signal stop if not already signaled
                          if not stop_event.is_set():
                              logger.warning(f"Target pages ({MAX_PAGES_TO_CRAWL}) reached before processing {final_url}. Signalling stop.")
                              stop_event.set()

                if should_add_result and text_content: # Only add if text was successfully extracted
                    page_title: str = final_url # Default title
                    soup = None
                    
                    # Create soup from raw_content only if HTML
                    if content_type == CONTENT_TYPE_HTML and raw_content:
                        try:
                            soup = BeautifulSoup(raw_content, 'html.parser')
                            title_tag = soup.find('title')
                            if title_tag and title_tag.string:
                                page_title = title_tag.string.strip()
                        except Exception as title_err:
                            logger.warning(f"Error extracting title for {final_url} via BS4: {title_err}.")
                    elif content_type == CONTENT_TYPE_PDF:
                         try: # Try getting filename for PDF title
                             pdf_filename = os.path.basename(urlparse(final_url).path)
                             if pdf_filename: page_title = pdf_filename
                         except Exception: pass

                    result_data: Dict[str, Any] = {
                        "source": content_type, # html, pdf
                        "url": final_url,
                        "title": page_title,
                        "content": f"{page_title}\n\n{text_content}", # Prepend title
                        "authors": [], # Placeholder metadata
                        "published": None, # Placeholder metadata
                    }
                    results_list.append(result_data)
                    PAGES_CRAWLED_TOTAL.inc() # Increment successful crawl counter
                    logger.info(f"Processed: {final_url} ({len(results_list)}/{MAX_PAGES_TO_CRAWL}) Type: {content_type}")

                    # Check if target reached AFTER adding result
                    if len(results_list) >= MAX_PAGES_TO_CRAWL:
                        if not stop_event.is_set():
                            logger.warning(f"Target pages ({MAX_PAGES_TO_CRAWL}) reached after processing {final_url}. Signalling stop.")
                            stop_event.set() # Signal other tasks to stop queuing

                # --- Process Discovered Links (Only if HTML, content processed, and not stopping) ---
                # Need soup object from title extraction part, check stop_event
                if content_type == CONTENT_TYPE_HTML and soup and not stop_event.is_set():
                    discovered_links: List[str] = []
                    base_url = final_url # Use final URL as base for relative links
                    # Use the soup object created earlier
                    # Search within main content tags preferentially, fallback to body
                    main_content_area = soup.find('article') or soup.find('main') or soup.body
                    if main_content_area:
                         for link_tag in main_content_area.find_all('a', href=True):
                            href = link_tag['href'].strip()
                            # Basic validation: not empty, not anchor, not javascript/mailto
                            if href and not href.startswith(('#', 'mailto:', 'javascript:')):
                                absolute_url = urljoin(base_url, href)
                                # Clean fragment identifiers (#...) from URL before queuing
                                absolute_url = urlparse(absolute_url)._replace(fragment="").geturl()
                                discovered_links.append(absolute_url)

                    # --- Queue Valid Links ---
                    if discovered_links:
                         logger.debug(f"Found {len(discovered_links)} potential links in {final_url}")
                         links_queued_this_page = 0
                         # Use a representative user agent for robots check
                         robot_user_agent = get_user_agent()

                         for link_url in discovered_links:
                             # Check stop event frequently during filtering
                             if stop_event.is_set(): break

                             # --- Link Filtering Logic ---
                             # 1. Visited Check (Bloom Filter)
                             if link_url in visited: continue

                             # 2. Scheme Check
                             if not link_url.startswith(('http://', 'https://')): continue

                             # 3. File Extension Check
                             try:
                                parsed_link_path = urlparse(link_url).path.lower()
                             except Exception:
                                logger.debug(f"Could not parse path for link {link_url}, skipping.")
                                continue # Skip unparseable URLs

                             skipped_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.zip', '.rar', '.tar.gz', '.pdf', # Also skip PDFs found via links for now
                                                  '.css', '.js', '.xml', '.json', '.txt', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.mp3', '.mp4', '.avi', '.mov', '.webm']
                             if any(parsed_link_path.endswith(ext) for ext in skipped_extensions):
                                logger.debug(f"Skipping link {link_url} due to extension.")
                                continue

                             # 4. Domain Filtering
                             link_domain = get_domain(link_url)
                             if not link_domain: continue # Skip if domain can't be extracted

                             domain_allowed = False
                             if ALLOWED_DOMAINS: # Strict allow list
                                 if any(link_domain == allowed or link_domain.endswith('.' + allowed) for allowed in ALLOWED_DOMAINS):
                                      domain_allowed = True
                             elif start_domains: # If no ALLOWED_DOMAINS, restrict to start domains
                                 if link_domain in start_domains:
                                      domain_allowed = True
                             # If neither ALLOWED_DOMAINS nor start_domains, domain_allowed remains False (no external links)

                             if not domain_allowed:
                                logger.debug(f"Skipping link {link_url} due to domain filtering (allowed: {ALLOWED_DOMAINS}, start: {start_domains}).")
                                continue

                            # 5. Robots.txt Check (Synchronous - potential bottleneck)
                             # Use final_url's domain as the base for finding robots.txt
                             # base_domain_url = f"{urlparse(final_url).scheme}://{urlparse(final_url).netloc}"
                             # if not _is_url_allowed_by_robots(base_domain_url, link_url, robot_user_agent):
                             #     logger.debug(f"Skipping {link_url} due to robots.txt disallowance on {base_domain_url}.")
                             #     continue

                             # --- Link Queuing ---
                             # Check visited set size limit
                             if len(visited) < visited_set_limit:
                                 # Check stop event again right before queuing
                                 if not stop_event.is_set():
                                     # visited.add(link_url) # Add happens at the start of _process_url now
                                     queue.append(link_url)
                                     links_queued_this_page += 1
                                 else: break # Stop queuing if event set
                             else:
                                 # Visited set limit reached globally
                                 if not stop_event.is_set():
                                     logger.warning(f"Visited set safety limit ({visited_set_limit}) reached. Signalling stop.")
                                     stop_event.set()
                                 break # Stop processing links for THIS page

                         if links_queued_this_page > 0:
                             logger.debug(f"Queued {links_queued_this_page} valid links from {final_url}")


            elif content_type == CONTENT_TYPE_FAILED:
                PAGES_FAILED_TOTAL.inc()
                logger.warning(f"Fetch/Parse failed for {current_url} (Final URL: {final_url}).")
            else: # Other type or HTML/PDF with no extracted text
                 # Only increment fail counter if no text was extracted for expected types
                if content_type in [CONTENT_TYPE_HTML, CONTENT_TYPE_PDF] and not text_content:
                     PAGES_FAILED_TOTAL.inc()
                # Log skipping other types if needed
                # logger.info(f"Skipped processing content for {final_url} (Type: {content_type})")


        except Exception as e:
            logger.error(f"Crawling/processing {current_url} failed unexpectedly in _process_url: {e}", exc_info=True)
            PAGES_FAILED_TOTAL.inc()
        # Semaphore is released automatically upon exiting the 'async with' block


async def _crawl_and_fetch_web_articles_async(
    start_urls: List[str],
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    proxies: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Core asynchronous crawl function."""
    results_list: List[Dict[str, Any]] = []
    # Increased capacity for better accuracy, adjust error rate as needed
    visited: BloomFilter = BloomFilter(capacity=max(MAX_PAGES_TO_CRAWL * 10, 5000), error_rate=0.001)
    queue: deque[str] = deque(start_urls)
    # Determine domains of start URLs for optional same-domain restriction
    start_domains: Set[Optional[str]] = {get_domain(url) for url in start_urls if get_domain(url)}
    # Safety limit for the visited set to prevent excessive memory usage with bloom filters
    visited_set_limit = MAX_PAGES_TO_CRAWL * 20 # Adjust multiplier as needed

    stop_event = asyncio.Event() # Event to signal stopping

    if proxies:
        logger.info(f"Using {len(proxies)} proxies for crawling.")
    else:
        logger.info("No proxies provided. Crawling directly.")

    # Process URLs until queue is empty or stop event is set
    while queue and not stop_event.is_set():
        # --- Batch Processing ---
        batch_size: int = min(len(queue), semaphore._value, 20) # Process up to N URLs concurrently, limited by semaphore & a max batch size
        tasks: List[asyncio.Task[Any]] = []
        urls_in_batch: List[str] = [] # Track URLs popped for this batch

        for _ in range(batch_size):
            if not queue or stop_event.is_set(): # Check queue and stop event
                break

            current_url: str = queue.popleft()
            urls_in_batch.append(current_url)

            # Select proxy for this specific task if needed
            proxy = random.choice(proxies) if proxies else None

            # Create the task using the _process_url defined outside the loop
            # Pass all necessary arguments
            task = asyncio.create_task(
                 _process_url(
                     current_url=current_url,
                     session=session,
                     results_list=results_list,
                     visited=visited,
                     queue=queue,
                     start_domains=start_domains,
                     visited_set_limit=visited_set_limit,
                     stop_event=stop_event,
                     semaphore=semaphore, # Pass the shared semaphore
                     proxy=proxy
                 )
             )
            tasks.append(task)

        # --- Wait for Batch Completion ---
        if tasks:
            logger.info(f"Starting batch of {len(tasks)} tasks. Queue: {len(queue)}, Results: {len(results_list)}")
            await asyncio.gather(*tasks)
            logger.info(f"Batch completed. Queue: {len(queue)}, Results: {len(results_list)}, Visited Est: ~{len(visited)}")
        else:
            # If no tasks were created (e.g., queue emptied or stop_event set while popping)
            if stop_event.is_set():
                logger.info("Stop event set, finishing crawl loop.")
                break
            # Add a small sleep if the queue was temporarily empty but stop not set, prevent busy-waiting
            if not queue:
                 logger.debug("Queue empty, sleeping briefly.")
                 await asyncio.sleep(0.5)


    # --- End of Main Crawl Loop ---
    logger.info(f"Crawling finished. Processed {len(results_list)} pages.")
    if len(queue) > 0 and len(results_list) >= MAX_PAGES_TO_CRAWL:
        logger.info(f"Note: Queue still contained {len(queue)} URLs when target page count met or stop triggered.")
    elif len(queue) == 0:
        logger.info("Processing queue is empty.")
    if stop_event.is_set():
         logger.info("Crawling stopped due to stop event (likely hit limits).")

    return results_list


async def crawl_and_fetch_web_articles(
    start_urls: List[str],
    proxies: Optional[List[str]] = None,
    max_concurrent: int = 10 # Max concurrent processing tasks
    ) -> List[Dict[str, Any]]:
    """
    Web crawler entry point: Sets up session, semaphore, and starts the async crawl.

    Args:
        start_urls: List of initial URLs to crawl.
        proxies: Optional list of proxy URLs (e.g., "http://user:pass@host:port").
        max_concurrent: Maximum number of URLs to process concurrently.

    Returns:
        List of dictionaries, each containing processed page data.
    """
    if not start_urls:
        logger.info("No starting URLs provided for crawling.")
        return []

    # Setup Semaphore based on desired concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    # Setup AIOHTTP session with connection limits
    connector = aiohttp.TCPConnector(limit_per_host=5, limit=max_concurrent * 2) # Fine-tune limits
    async with aiohttp.ClientSession(connector=connector) as session:
        return await _crawl_and_fetch_web_articles_async(
            start_urls=start_urls,
            session=session,
            semaphore=semaphore,
            proxies=proxies
        )


# --- arXiv Fetcher ---
def fetch_arxiv_papers(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Fetches paper metadata and abstracts from arXiv using its API."""
    logger.info(f"Querying arXiv for '{query}' (max: {max_results})")
    if max_results <= 0: return []

    # Configure the arXiv client
    client = arxiv.Client(
        page_size = min(max_results, 500), # Request larger pages if fetching many results
        delay_seconds = 5, # Be polite to the API
        num_retries = 3
    )

    # Build the search query
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance # Or Relevance, LastUpdatedDate
    )

    results_list: List[Dict[str, Any]] = []
    fetched_count = 0

    try:
        # Use client.results for automatic handling of pagination and retries
        results_generator = client.results(search)

        for result in results_generator:
            # Double-check against max_results needed as client might fetch full page
            if fetched_count >= max_results:
                 logger.info(f"Reached max_results ({max_results}) for arXiv query.")
                 break

            # Clean up summary: replace newlines, strip whitespace
            clean_summary = "No abstract available."
            if result.summary:
                 clean_summary = ' '.join(result.summary.splitlines()).strip()

            title = (result.title or "Untitled").strip()

            # Combine title and summary for content, handle missing summary
            content = f"{title}. {clean_summary}"

            # Get authors as a list of strings
            authors = [str(author) for author in result.authors] if result.authors else []

            # Format published date
            published_date = result.published.isoformat() if result.published else None
            entry_id = result.entry_id # URL like http://arxiv.org/abs/2303.08774v4

            # Prefer PDF URL if available, otherwise use the abstract page URL
            paper_url = result.pdf_url if result.pdf_url else entry_id

            paper_data = {
                "source": CONTENT_TYPE_ARXIV, # Use constant
                "entry_id": entry_id,
                "title": title,
                "content": content,
                "authors": authors,
                "published": published_date,
                "url": paper_url
            }
            results_list.append(paper_data)
            fetched_count += 1

        logger.info(f"Fetched {len(results_list)} papers from arXiv for query '{query}'.")

    except Exception as e:
        # Log specific arXiv exceptions differently? Maybe later.
        logger.error(f"An error occurred while fetching from arXiv: {e}", exc_info=True)

    finally:
        # No explicit client cleanup needed for the default client
        return results_list


# --- Example Usage (Optional) ---
async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # Example 1: Fetch arXiv papers
    arxiv_query = "large language model retrieval augmented generation"
    arxiv_results = fetch_arxiv_papers(query=arxiv_query, max_results=5)
    print(f"\n--- arXiv Results ({len(arxiv_results)}) ---")
    for paper in arxiv_results:
        print(f"Title: {paper['title']}")
        print(f"URL: {paper['url']}")
        print(f"Published: {paper['published']}")
        # print(f"Abstract Snippet: {paper['content'][:200]}...") # Content includes title now
        print("-" * 10)

    # Example 2: Crawl web pages
    # start_web_urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/"] # Example blog post
    start_web_urls = ["https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/"] # Example technical page
    # Optional: Add proxies like ["http://user:pass@host1:port", "http://user:pass@host2:port"]
    proxies_list = None

    # Limit crawl for example
    global MAX_PAGES_TO_CRAWL
    MAX_PAGES_TO_CRAWL = 5 # Override config for testing

    web_results = await crawl_and_fetch_web_articles(
        start_urls=start_web_urls,
        proxies=proxies_list,
        max_concurrent=5 # Limit concurrency for example
    )
    print(f"\n--- Web Crawl Results ({len(web_results)}) ---")
    for page in web_results:
        print(f"Title: {page['title']}")
        print(f"URL: {page['url']}")
        print(f"Type: {page['source']}")
        # print(f"Content Snippet: {page['content'][:300]}...") # Content includes title now
        print("-" * 10)

if __name__ == "__main__":
    asyncio.run(main())