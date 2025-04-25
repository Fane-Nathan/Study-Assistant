# -*- coding: utf-8 -*-
"""
Fetches and processes data from arXiv and web sources (HTML/PDF) asynchronously.
Refactored for improved modularity and readability.
Uses Playwright to handle dynamic websites and aiohttp for direct PDF downloads.
Processes discovered PDF links in parallel tasks.
Includes safe unpacking for link extraction results.
"""

import asyncio
import io
import logging
import os
import random
import re
import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Set
from urllib.parse import urljoin, urlparse, urlunparse

import aiohttp # Keep for direct PDF fetching
import arxiv
import fitz # PyMuPDF
from bs4 import BeautifulSoup
from cachetools import LRUCache
from pybloom_live import BloomFilter
from trafilatura import extract
from trafilatura.settings import use_config

# --- Playwright Import ---
try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError, Error as PlaywrightError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    logging.getLogger(__name__).critical("Playwright not found. Please install it: pip install playwright && playwright install")
    PLAYWRIGHT_AVAILABLE = False


logger = logging.getLogger(__name__)

# --- Configuration Loading ---
try:
    from .. import config
    FETCH_TIMEOUT = config.FETCH_TIMEOUT * 1000 # Playwright uses milliseconds
    AIOHTTP_FETCH_TIMEOUT = config.FETCH_TIMEOUT # aiohttp uses seconds
    HEAD_TIMEOUT = config.HEAD_TIMEOUT # aiohttp uses seconds
    MAX_PAGES_TO_CRAWL = config.MAX_PAGES_TO_CRAWL
    CRAWL_DELAY_SECONDS = config.CRAWL_DELAY_SECONDS
    ALLOWED_DOMAINS = config.ALLOWED_DOMAINS
except ImportError:
    logger.warning("config.py not found or variables missing. Using default values.")
    FETCH_TIMEOUT = 30000 # ms (for Playwright)
    AIOHTTP_FETCH_TIMEOUT = 30 # s (for aiohttp)
    HEAD_TIMEOUT = 15 # s (for aiohttp)
    MAX_PAGES_TO_CRAWL = 100
    CRAWL_DELAY_SECONDS = 1.0
    ALLOWED_DOMAINS = None

# --- Constants & Settings ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
]
SKIPPED_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp',
    '.zip', '.rar', '.tar.gz', '.7z',
    '.pdf', # Skip adding PDF links to the *crawling queue*
    '.css', '.js',
    '.xml', '.json', '.txt',
    '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.mp3', '.wav', '.ogg',
    '.mp4', '.avi', '.mov', '.wmv', '.webm',
    '.exe', '.dmg', '.iso',
]
CONTENT_TYPE_HTML = "html"
CONTENT_TYPE_PDF = "pdf"
CONTENT_TYPE_OTHER = "other"
CONTENT_TYPE_FAILED = "failed"
CONTENT_TYPE_ARXIV = "arxiv"

# --- State Variables ---
LAST_REQUEST_TIME: float = 0.0 # Global for simple rate limiting

# --- Caching ---
CACHE_MAX_SIZE = 50
TRAFILATURA_CONFIG = use_config()
TRAFILATURA_CONFIG.set("DEFAULT", "EXTRACTION_TIMEOUT", "60")
# Cache stores: (final_url, content_type, text_content)
resource_cache: LRUCache = LRUCache(maxsize=CACHE_MAX_SIZE)

# --- Helper Functions ---
# get_user_agent, get_domain, _clean_text remain the same

def get_user_agent() -> str:
    """Selects a random User-Agent string."""
    if not USER_AGENTS:
        return "Mozilla/5.0 (compatible; Python Fetcher)"
    return random.choice(USER_AGENTS)

def get_domain(url: str) -> Optional[str]:
    """Extracts the 'netloc' (e.g., 'google.com') from a URL."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        logger.warning(f"Could not parse domain from URL: {url}")
        return None

def _clean_text(text: Optional[str]) -> Optional[str]:
    """Cleans extracted text by normalizing whitespace and reducing excessive newlines."""
    if not text: return None
    try:
        text = re.sub(r'[ \t]+', ' ', text) # Normalize spaces/tabs
        lines = text.splitlines()
        cleaned_lines = [line.strip() for line in lines if line.strip()] # Remove empty lines and strip
        if not cleaned_lines: return None
        text = "\n".join(cleaned_lines) # Join with single newline
        text = re.sub(r'\n{3,}', '\n\n', text) # Reduce excessive newlines
        cleaned_text = text.strip() # Final strip
        return cleaned_text if cleaned_text else None
    except Exception as e:
        logger.error(f"Text cleaning failed: {e}")
        return text # Return original on error


# --- Content Parsing Functions ---
# _parse_pdf_content remains the same
def _parse_pdf_content(pdf_bytes: bytes, source_url: str) -> Optional[str]:
    """Parses text content from PDF bytes."""
    pdf_text = ""
    try:
        with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
            for page in doc:
                pdf_text += page.get_text("text", sort=True) + "\n\n" # Added newline between pages
        cleaned_text = _clean_text(pdf_text)
        if not cleaned_text:
            logger.warning(f"No text extracted from PDF after cleaning: {source_url}")
        return cleaned_text
    except Exception as e:
        logger.error(f"PDF processing failed for {source_url}: {e}", exc_info=True)
        return None

# _parse_html_content remains the same
def _parse_html_content(html_string: str, source_url: str) -> Optional[str]:
    """Parses text content from HTML string using Trafilatura with BS4 fallback."""
    cleaned_text: Optional[str] = None
    try:
        # 1. Attempt extraction with Trafilatura
        extracted_text_tf: Optional[str] = None
        try:
            # Trafilatura works better with HTML string
            extracted_text_tf = extract(html_string, config=TRAFILATURA_CONFIG, favor_recall=True,
                                        include_comments=False, include_tables=True, url=source_url)
            if extracted_text_tf:
                logger.debug(f"Extracted text using Trafilatura for {source_url}")
            else:
                logger.debug(f"Trafilatura extracted no text for {source_url}, will try fallback.")
        except Exception as extraction_err:
            logger.warning(f"Trafilatura failed for {source_url}: {extraction_err}. Falling back.")

        cleaned_text = _clean_text(extracted_text_tf)

        # 2. Fallback to basic BeautifulSoup if needed
        if not cleaned_text:
            logger.info(f"Falling back to basic BS4 text extraction for {source_url}")
            soup_fallback = BeautifulSoup(html_string, 'html.parser')
            main_content_area = soup_fallback.find('article') or soup_fallback.find('main') or soup_fallback.body
            if main_content_area:
                raw_text_fallback: str = main_content_area.get_text(separator='\n', strip=True)
                cleaned_text = _clean_text(raw_text_fallback)
                if cleaned_text:
                    logger.debug(f"Extracted text using BS4 fallback for {source_url}")
            if not cleaned_text:
                logger.warning(f"Could not extract text using BS4 fallback for {source_url}")

        return cleaned_text
    except Exception as e:
        logger.error(f"HTML parsing failed unexpectedly for {source_url}: {e}", exc_info=True)
        return None

# --- aiohttp Fetching Logic ---
# _fetch_url_content_aiohttp remains the same
async def _fetch_url_content_aiohttp(url: str, session: aiohttp.ClientSession, proxy: Optional[str] = None) -> Tuple[str, str, Optional[bytes]]:
    """
    Performs HEAD and GET requests using aiohttp to fetch content type and raw bytes.
    Handles redirects and basic HTTP errors. Intended for direct file downloads like PDFs.

    Returns:
        Tuple[str, str, Optional[bytes]]: (final_url, content_type_constant, raw_content_bytes)
    """
    final_url = url
    headers = {'User-Agent': get_user_agent()}
    content_type_header = ""
    # HEAD Request first to check type without downloading full content
    try:
        logger.debug(f"Sending HEAD request (aiohttp) to {url}")
        async with session.head(url, headers=headers, timeout=HEAD_TIMEOUT, allow_redirects=True, proxy=proxy) as head_response:
            head_response.raise_for_status()
            content_type_header = head_response.headers.get("Content-Type", "").lower()
            final_url = str(head_response.url)
            logger.debug(f"HEAD success (aiohttp) for {url}. Final URL: {final_url}, Content-Type: {content_type_header}")
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.warning(f"HEAD request failed (aiohttp) for {url}: {e}. Aborting fetch.")
        return url, CONTENT_TYPE_FAILED, None
    except Exception as e:
        logger.error(f"Unexpected error during HEAD request (aiohttp) for {url}: {e}", exc_info=True)
        return url, CONTENT_TYPE_FAILED, None

    # Determine Type and Fetch Content (GET Request)
    is_pdf = 'application/pdf' in content_type_header
    is_html = 'text/html' in content_type_header # Less likely needed here, but check anyway

    if is_pdf or is_html: # Primarily interested in PDFs here
        try:
            logger.debug(f"Sending GET request (aiohttp) to {final_url}")
            async with session.get(final_url, headers=headers, timeout=AIOHTTP_FETCH_TIMEOUT, allow_redirects=True, proxy=proxy) as response:
                response.raise_for_status()
                final_url = str(response.url)
                raw_content = await response.read()
                logger.debug(f"GET success (aiohttp) for {final_url}. Read {len(raw_content)} bytes.")
                content_type = CONTENT_TYPE_PDF if is_pdf else CONTENT_TYPE_HTML
                return final_url, content_type, raw_content
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"GET request failed (aiohttp) for {final_url}: {e}")
            return final_url, CONTENT_TYPE_FAILED, None
        except Exception as e:
             logger.error(f"Unexpected error during GET request (aiohttp) for {final_url}: {e}", exc_info=True)
             return final_url, CONTENT_TYPE_FAILED, None
    else:
         logger.info(f"Skipping content fetch (aiohttp) for non-PDF/HTML type '{content_type_header}' at {final_url}")
         return final_url, CONTENT_TYPE_OTHER, None


# --- Playwright Fetching Logic ---
# _fetch_with_playwright remains the same
async def _fetch_with_playwright(url: str, playwright_context) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Fetches an HTML URL using Playwright, waits for dynamic content.

    Returns:
        Tuple[str, str, Optional[str], Optional[str]]:
        (final_url, content_type_constant, html_content_string, error_message)
        Returns CONTENT_TYPE_FAILED or CONTENT_TYPE_OTHER if fetch fails or type is not HTML.
    """
    page = None
    final_url = url
    error_message = None
    html_content = None
    content_type = CONTENT_TYPE_FAILED # Default to failed

    try:
        page = await playwright_context.new_page()
        await page.set_extra_http_headers({'User-Agent': get_user_agent()})

        logger.debug(f"Navigating to {url} with Playwright...")
        response = await page.goto(url, timeout=FETCH_TIMEOUT, wait_until="domcontentloaded")
        final_url = page.url

        if response is None: raise PlaywrightError(f"Navigation to {url} returned None response.")
        status = response.status
        if not 200 <= status < 300: raise PlaywrightError(f"HTTP Error {status} for {final_url}")

        content_type_header = response.headers.get("content-type", "").lower()
        logger.debug(f"Playwright navigation success for {url}. Final URL: {final_url}, Status: {status}, Content-Type: {content_type_header}")

        # --- Wait for dynamic content specifically for OpenReview ---
        wait_timeout = 15000 # ms
        try:
            # Increased specificity: Wait for the note container AND a link inside it
            await page.wait_for_selector('.note-list .note a[href*="id="]', timeout=wait_timeout)
            logger.info(f"Detected note list items with links on {final_url}.")
        except PlaywrightTimeoutError:
             logger.warning(f"Did not find '.note-list .note a[href*=\"id=\"]' selector within {wait_timeout}ms on {final_url}. Content might be incomplete.")
        # Fallback: wait for network idle briefly
        try:
             await page.wait_for_load_state('networkidle', timeout=5000)
             logger.debug(f"Network idle state reached for {final_url}")
        except PlaywrightTimeoutError:
             logger.warning(f"Network did not become idle for {final_url} after note check, proceeding anyway.")

        # --- Get Content if HTML ---
        if 'text/html' in content_type_header:
            html_content = await page.content()
            content_type = CONTENT_TYPE_HTML
            logger.debug(f"Fetched HTML content ({len(html_content)} chars) for {final_url}")
        else:
            content_type = CONTENT_TYPE_OTHER # Or determine if PDF based on header if needed
            logger.info(f"Skipping content fetch for non-HTML type '{content_type_header}' at {final_url} in Playwright fetch.")

    except PlaywrightTimeoutError as e:
        error_message = f"Playwright TimeoutError for {url}: {e}"
        logger.error(error_message)
    except PlaywrightError as e:
        error_message = f"Playwright Error for {url}: {e}"
        logger.error(error_message)
    except Exception as e:
        error_message = f"Unexpected error during Playwright fetch for {url}: {e}"
        logger.error(error_message, exc_info=True)
    finally:
        if page:
             try: await page.close()
             except Exception as close_err: logger.warning(f"Error closing Playwright page for {url}: {close_err}")


    return final_url, content_type, html_content, error_message


# --- Parsing Orchestrator ---
# _get_parsed_content_orchestrator remains the same
async def _get_parsed_content_orchestrator(
    url: str,
    playwright_context,
    session: aiohttp.ClientSession, # Keep for direct PDF fetch
    is_pdf_link: bool, # Flag to indicate if we know it's likely a PDF
    proxy: Optional[str] = None
) -> Tuple[str, str, Optional[str]]:
    """
    Orchestrates fetching and parsing. Uses Playwright for HTML, aiohttp for PDFs.

    Returns:
        Tuple[str, str, Optional[str]]: (final_url, content_type_constant, text_content)
    """
    final_url = url
    content_type = CONTENT_TYPE_FAILED
    cleaned_text: Optional[str] = None

    if is_pdf_link:
        # Directly fetch PDF using aiohttp
        logger.info(f"Attempting direct download of potential PDF: {url}")
        try:
            final_url, content_type, pdf_bytes = await _fetch_url_content_aiohttp(url, session, proxy)
            if content_type == CONTENT_TYPE_PDF and pdf_bytes:
                cleaned_text = _parse_pdf_content(pdf_bytes, final_url)
            elif content_type != CONTENT_TYPE_FAILED:
                 logger.warning(f"Expected PDF but got {content_type} for {url}")
                 content_type = CONTENT_TYPE_OTHER # Mark as other if not PDF or failed
            # If failed, content_type remains CONTENT_TYPE_FAILED
        except Exception as pdf_err:
            logger.error(f"Error directly fetching/parsing PDF {url}: {pdf_err}", exc_info=True)
            content_type = CONTENT_TYPE_FAILED # Ensure failure state
    else:
        # Fetch HTML using Playwright
        try:
            final_url, content_type, html_content, error_msg = await _fetch_with_playwright(url, playwright_context)
            if error_msg:
                 content_type = CONTENT_TYPE_FAILED
            elif content_type == CONTENT_TYPE_HTML and html_content:
                 cleaned_text = _parse_html_content(html_content, final_url)
            # Other content types (like OTHER) will result in cleaned_text being None
        except Exception as html_err:
             logger.error(f"Error fetching/parsing HTML {url} with Playwright: {html_err}", exc_info=True)
             content_type = CONTENT_TYPE_FAILED

    # No caching implemented here for simplicity with Playwright
    return final_url, content_type, cleaned_text


# --- Link Extraction and Filtering ---
# _extract_links and _is_link_valid_for_queueing remain the same

def _extract_links(html_string: str, base_url: str) -> List[Tuple[str, str]]:
    """
    Extracts potential navigation links from HTML string.
    Returns a list of tuples: (absolute_url_no_fragment, link_text)
    """
    discovered_links: List[Tuple[str, str]] = []
    try:
        soup = BeautifulSoup(html_string, 'html.parser')
        # --- Specific Selector for OpenReview PDF links ---
        pdf_links = soup.find_all('a', href=re.compile(r'^/(pdf|attachment)\?id='))
        logger.debug(f"Found {len(pdf_links)} potential PDF links using specific pattern on {base_url}")
        for link_tag in pdf_links:
             href = link_tag['href'].strip()
             # Try to get paper title from parent structure more robustly
             link_text = link_tag.get_text(strip=True) # Default to link's own text
             parent_note = link_tag.find_parent('div', class_='note')
             if parent_note:
                  title_tag = parent_note.find('h4') # Find h4 within the parent note
                  if title_tag:
                       link_text = title_tag.get_text(strip=True)

             try:
                 absolute_url = urljoin(base_url, href)
                 parsed_abs_url = urlparse(absolute_url)
                 absolute_url_no_frag = urlunparse(parsed_abs_url._replace(fragment="")) # Use urlunparse
                 discovered_links.append((absolute_url_no_frag, link_text)) # Appends a tuple of 2 items
             except ValueError: logger.debug(f"Skipping invalid relative URL '{href}' found on {base_url}")

    except Exception as e:
        logger.error(f"Error extracting links from {base_url}: {e}", exc_info=True)
    return discovered_links

def _is_link_valid_for_queueing(
    link_url: str,
    visited: BloomFilter,
    start_domains: Set[Optional[str]],
    ) -> bool:
    """Applies filtering rules to a discovered link to decide if it should be CRAWLED."""
    # 1. Visited Check
    if link_url in visited: return False

    # 2. Scheme Check
    if not link_url.startswith(('http://', 'https://')): return False

    # 3. File Extension Check (Strictly for QUEUEING - PDFs handled separately)
    try:
        parsed_link_path = urlparse(link_url).path.lower()
        # Also check OpenReview patterns
        if parsed_link_path.startswith(('/pdf', '/attachment')):
             return False
    except Exception:
        logger.debug(f"Could not parse path for link {link_url}, skipping queue check.")
        return False
    if any(parsed_link_path.endswith(ext) for ext in SKIPPED_EXTENSIONS):
        # logger.debug(f"Skipping link {link_url} for QUEUEING due to extension.")
        return False

    # 4. Domain Filtering
    link_domain = get_domain(link_url)
    if not link_domain: return False
    domain_allowed = False
    if ALLOWED_DOMAINS: # Explicit allow list takes precedence
        if any(link_domain == allowed or link_domain.endswith('.' + allowed) for allowed in ALLOWED_DOMAINS):
            domain_allowed = True
    elif start_domains: # Fallback to same-domain policy if no explicit list
        if link_domain in start_domains:
            domain_allowed = True
    # If neither is set, domain_allowed remains False (no external links followed)
    if not domain_allowed:
        # logger.debug(f"Skipping link {link_url} for QUEUEING due to domain filtering.")
        return False

    return True # Link passed all filters FOR QUEUEING


# --- Dedicated PDF Processing Task ---
# _process_pdf_link_task remains the same
async def _process_pdf_link_task(
    pdf_url: str,
    link_text: str, # Title hint from the link
    found_on_url: str, # URL where the link was found
    session: aiohttp.ClientSession,
    results_list: List[Dict[str, Any]],
    visited: BloomFilter, # Pass the main visited filter
    stop_event: asyncio.Event,
    semaphore: asyncio.Semaphore,
    proxy: Optional[str] = None
):
    """Fetches, parses, and adds a single PDF link result."""
    global LAST_REQUEST_TIME
    if stop_event.is_set() or pdf_url in visited:
        return

    async with semaphore:
        if stop_event.is_set() or pdf_url in visited: # Double check after acquiring semaphore
            return
        visited.add(pdf_url) # Mark as visited *before* processing

        # Rate limiting for PDF downloads
        loop = asyncio.get_event_loop()
        now = loop.time()
        time_since_last = now - LAST_REQUEST_TIME
        if time_since_last < CRAWL_DELAY_SECONDS:
            await asyncio.sleep(CRAWL_DELAY_SECONDS - time_since_last)
        LAST_REQUEST_TIME = loop.time()

        logger.debug(f"Starting PDF processing task for: {pdf_url}")
        try:
            # Fetch and parse PDF using aiohttp directly
            final_pdf_url, pdf_content_type, pdf_bytes = await _fetch_url_content_aiohttp(pdf_url, session, proxy)

            if pdf_content_type == CONTENT_TYPE_PDF and pdf_bytes:
                pdf_text_content = _parse_pdf_content(pdf_bytes, final_pdf_url)
                if pdf_text_content:
                    # Add PDF result if within limits
                    if len(results_list) < MAX_PAGES_TO_CRAWL:
                        pdf_title = link_text if link_text else os.path.basename(urlparse(final_pdf_url).path)
                        if not pdf_title: pdf_title = final_pdf_url

                        pdf_result_data = {"source": CONTENT_TYPE_PDF, "url": final_pdf_url, "title": pdf_title,
                                           "content": f"{pdf_title}\n\n{pdf_text_content}",
                                           "authors": [], "published": None, "found_on": found_on_url}
                        results_list.append(pdf_result_data)
                        logger.info(f"Processed PDF Link: {final_pdf_url} ({len(results_list)}/{MAX_PAGES_TO_CRAWL}) Found on: {found_on_url}")

                        # Check limit again AFTER adding PDF
                        if len(results_list) >= MAX_PAGES_TO_CRAWL and not stop_event.is_set():
                            logger.warning(f"Target pages ({MAX_PAGES_TO_CRAWL}) reached after processing PDF {final_pdf_url}. Signalling stop.")
                            stop_event.set()
                    else: # Hit limit
                         if not stop_event.is_set(): stop_event.set()
                else:
                    logger.warning(f"Failed to parse PDF content from: {final_pdf_url}")

            elif pdf_content_type == CONTENT_TYPE_FAILED:
                 logger.warning(f"Failed to fetch PDF link: {pdf_url}")
            else:
                 logger.warning(f"Expected PDF but got {pdf_content_type} for link: {pdf_url}")

        except Exception as pdf_err:
            logger.error(f"Error processing PDF link task for {pdf_url}: {pdf_err}", exc_info=True)


# --- Web Crawler Task (Modified) ---
async def _process_crawl_url_pw(
    current_url: str,
    playwright_context,
    session: aiohttp.ClientSession,
    results_list: List[Dict[str, Any]],
    visited: BloomFilter,
    queue: deque[str],
    active_tasks: Set[asyncio.Task],
    start_domains: Set[Optional[str]],
    visited_set_limit: int,
    stop_event: asyncio.Event,
    semaphore: asyncio.Semaphore,
    process_pdfs: bool,
    proxy: Optional[str] = None
):
    """
    Task function using Playwright for initial HTML fetch and dynamic content.
    Spawns separate tasks for processing discovered PDF links.
    """
    global LAST_REQUEST_TIME

    if stop_event.is_set() or current_url in visited: return

    # --- Rate Limiting for HTML Page Fetch ---
    loop = asyncio.get_event_loop()
    now = loop.time()
    time_since_last = now - LAST_REQUEST_TIME
    if time_since_last < CRAWL_DELAY_SECONDS:
        await asyncio.sleep(CRAWL_DELAY_SECONDS - time_since_last)
    LAST_REQUEST_TIME = loop.time()
    # ---

    visited.add(current_url)

    # 2. Fetch and Parse HTML using Playwright
    html_content_for_links: Optional[str] = None
    text_content: Optional[str] = None
    final_url = current_url
    content_type = CONTENT_TYPE_FAILED

    try:
        final_url, content_type, html_content_for_links, error_msg = await _fetch_with_playwright(url=current_url, playwright_context=playwright_context)

        if final_url != current_url:
             if final_url in visited: return
             visited.add(final_url)

        if error_msg:
            logger.warning(f"Playwright fetch failed for {current_url}: {error_msg}")
            return

        if content_type == CONTENT_TYPE_HTML and html_content_for_links:
            text_content = _parse_html_content(html_content_for_links, final_url)

            # 3. Add HTML Result (if successful)
            if text_content:
                 if len(results_list) < MAX_PAGES_TO_CRAWL:
                     page_title = final_url
                     try:
                         soup_title = BeautifulSoup(html_content_for_links, 'html.parser')
                         title_tag = soup_title.find('title')
                         if title_tag and title_tag.string: page_title = title_tag.string.strip()
                     except Exception: pass
                     if not page_title or page_title == final_url:
                          first_line = text_content.split('\n', 1)[0].strip()
                          if first_line and len(first_line) < 100: page_title = first_line

                     result_data = {"source": content_type, "url": final_url, "title": page_title,
                                    "content": text_content, "authors": [], "published": None}
                     results_list.append(result_data)
                     logger.info(f"Processed HTML: {final_url} ({len(results_list)}/{MAX_PAGES_TO_CRAWL})")
                     if len(results_list) >= MAX_PAGES_TO_CRAWL and not stop_event.is_set():
                         logger.warning(f"Target pages ({MAX_PAGES_TO_CRAWL}) reached. Signalling stop.")
                         stop_event.set()
                 else:
                      if not stop_event.is_set(): stop_event.set()
            else:
                 logger.warning(f"Failed to parse text content from HTML: {final_url}")


        # 4. Extract Links and Spawn PDF Tasks / Queue HTML Links
        if html_content_for_links and not stop_event.is_set():
            discovered_links = _extract_links(html_content_for_links, final_url)
            links_queued_this_page = 0
            pdfs_spawned_this_page = 0

            # --- DEBUGGING: Inspect discovered_links before the loop ---
            logger.debug(f"Inspecting links found on {final_url}. Total: {len(discovered_links)}")
            if discovered_links:
                 # Safely log the first item if the list is not empty
                 logger.debug(f"First link item type: {type(discovered_links[0])}, value: {repr(discovered_links[0])[:200]}")
                 all_tuples_len_2 = all(isinstance(item, tuple) and len(item) == 2 for item in discovered_links)
                 logger.debug(f"All items are 2-element tuples: {all_tuples_len_2}")
                 if not all_tuples_len_2:
                      for i, item in enumerate(discovered_links):
                          if not (isinstance(item, tuple) and len(item) == 2):
                              logger.warning(f"Problematic item at index {i} on {final_url}: type={type(item)}, value={repr(item)[:200]}")
            # --- END DEBUGGING ---

            for link_data in discovered_links: # Iterate safely first
                 if stop_event.is_set(): break

                 # --- FIXED: Safely unpack the tuple ---
                 if isinstance(link_data, tuple) and len(link_data) == 2:
                     link_url, link_text = link_data # Unpack only if it's a valid pair
                 else:
                     # Log a warning if the item is malformed and skip it
                     logger.warning(f"Skipping malformed link data found on {final_url}: {repr(link_data)[:200]}")
                     continue # Go to the next item in discovered_links
                 # --- End Fix ---

                 parsed_link = urlparse(link_url)
                 link_path_lower = parsed_link.path.lower()
                 is_potential_pdf = link_path_lower.endswith(".pdf") or link_path_lower.startswith(('/pdf', '/attachment'))

                 # --- Spawn PDF Task ---
                 if process_pdfs and is_potential_pdf:
                     if link_url not in visited:
                         if len(visited) >= visited_set_limit:
                             if not stop_event.is_set(): stop_event.set()
                             break

                         logger.debug(f"Spawning PDF task for: {link_url} (found on {final_url})")
                         pdf_task = asyncio.create_task(
                             _process_pdf_link_task(
                                 pdf_url=link_url, link_text=link_text, found_on_url=final_url,
                                 session=session, results_list=results_list, visited=visited,
                                 stop_event=stop_event, semaphore=semaphore, proxy=proxy
                             )
                         )
                         active_tasks.add(pdf_task)
                         pdf_task.add_done_callback(active_tasks.discard)
                         pdfs_spawned_this_page += 1
                 # --- END Spawn PDF Task ---

                 # --- ELSE: Queue HTML Link ---
                 elif not is_potential_pdf:
                     if len(visited) >= visited_set_limit:
                         if not stop_event.is_set(): stop_event.set()
                         break
                     if _is_link_valid_for_queueing(link_url, visited, start_domains):
                         queue.append(link_url)
                         links_queued_this_page += 1
                 # --- END ELSE ---

            # Log summary for the page
            if links_queued_this_page > 0: logger.debug(f"Queued {links_queued_this_page} valid HTML links from {final_url}")
            if pdfs_spawned_this_page > 0: logger.debug(f"Spawned {pdfs_spawned_this_page} PDF processing tasks from {final_url}")

    except Exception as e:
        logger.error(f"Processing {current_url} failed unexpectedly in task: {e}", exc_info=True)


# --- Main Crawler Orchestration ---
# _crawl_manager_pw remains the same
async def _crawl_manager_pw(
    start_urls: List[str],
    playwright_context,
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    process_pdfs: bool,
    proxies: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Manages the crawling process using Playwright tasks."""
    results_list: List[Dict[str, Any]] = []
    estimated_capacity = MAX_PAGES_TO_CRAWL * (40 if process_pdfs else 15) # Increased estimate
    visited: BloomFilter = BloomFilter(capacity=max(estimated_capacity, 10000), error_rate=0.001)
    queue: deque[str] = deque(start_urls)
    start_domains: Set[Optional[str]] = {get_domain(url) for url in start_urls if get_domain(url)}
    visited_set_limit = visited.capacity
    stop_event = asyncio.Event()
    active_tasks = set() # Keep track of all active tasks (HTML + PDF)

    logger.info(f"Crawl Manager started. Queue size: {len(queue)}. Visited capacity: {visited.capacity}")

    while (queue or active_tasks) and not stop_event.is_set():
        # Launch new HTML processing tasks from queue
        while queue and len(active_tasks) < semaphore._value and not stop_event.is_set():
            current_url = queue.popleft()
            # Check visited *before* creating task
            if current_url in visited:
                continue

            # Acquire semaphore *before* creating task for HTML page processing
            await semaphore.acquire()
            logger.debug(f"Acquired semaphore for HTML task: {current_url}. Current tasks: {len(active_tasks)}")

            proxy = random.choice(proxies) if proxies else None
            task = asyncio.create_task(
                _process_crawl_url_pw( # Use the Playwright task function
                    current_url=current_url,
                    playwright_context=playwright_context,
                    session=session,
                    results_list=results_list,
                    visited=visited, queue=queue,
                    active_tasks=active_tasks, # Pass task set
                    start_domains=start_domains,
                    visited_set_limit=visited_set_limit, stop_event=stop_event,
                    semaphore=semaphore, process_pdfs=process_pdfs,
                    proxy=proxy
                )
            )
            active_tasks.add(task)
            # Release semaphore ONLY when the HTML task itself completes
            # The PDF tasks spawned within will manage the semaphore independently
            # --- FIXED: Ensure semaphore is released even if task errors ---
            def done_callback(fut, url=current_url):
                logger.debug(f"HTML task for {url} finished. Releasing semaphore. Current tasks: {len(active_tasks)}")
                semaphore.release()
                active_tasks.discard(fut)
                # Log potential exceptions from the task
                try:
                    fut.result() # Check for exceptions
                except asyncio.CancelledError:
                     logger.debug(f"HTML task for {url} was cancelled.")
                except Exception as task_exc:
                     logger.error(f"HTML task for {url} raised an exception: {task_exc}", exc_info=False) # Avoid excessive traceback logging here

            task.add_done_callback(done_callback)
            # --- End Fix ---


        # Wait for any task (HTML or PDF) to complete
        if not active_tasks:
            if queue and not stop_event.is_set(): # Should not happen if semaphore logic is right
                 logger.warning("No active tasks but queue is not empty? Continuing loop.")
                 await asyncio.sleep(0.1)
                 continue
            else: # Queue is empty and no tasks running
                 break

        logger.debug(f"Waiting for tasks to complete. Active: {len(active_tasks)}, Queue: {len(queue)}")
        done, pending = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
        logger.debug(f"Tasks completed: {len(done)}. Pending: {len(pending)}.")
        # Tasks are automatically removed from active_tasks by their own done_callbacks now

        # Small sleep to prevent tight loop if tasks complete very quickly
        await asyncio.sleep(0.05)


    # --- Cleanup after loop ---
    if active_tasks:
        logger.info(f"Crawling loop finished or stopped. Cancelling {len(active_tasks)} remaining tasks...")
        for task in list(active_tasks): # Iterate over a copy
            task.cancel()
        # Wait for cancellations with a timeout
        try:
             await asyncio.wait_for(asyncio.gather(*active_tasks, return_exceptions=True), timeout=10.0)
             logger.info("Remaining tasks cancelled.")
        except asyncio.TimeoutError:
             logger.warning("Timeout waiting for remaining tasks to cancel.")
        except Exception as gather_err:
             logger.error(f"Error during final task gather: {gather_err}")


    logger.info(f"Crawling finished. Processed {len(results_list)} total items (HTML pages + PDFs).")
    if len(queue) > 0: logger.info(f"Note: Queue still contained {len(queue)} URLs when crawl ended.")
    if stop_event.is_set(): logger.info("Crawling stopped due to stop event.")
    return results_list


# --- Public API Function ---
# crawl_and_fetch_web_articles remains the same
async def crawl_and_fetch_web_articles(
    start_urls: List[str],
    process_pdfs_linked: bool = True,
    max_pages_override: Optional[int] = None,
    proxies: Optional[List[str]] = None,
    max_concurrent: int = 10 # Increased default concurrency slightly
    ) -> List[Dict[str, Any]]:
    """
    Public entry point for web crawling using Playwright for dynamic content.
    """
    if not PLAYWRIGHT_AVAILABLE:
        logger.error("Playwright is not installed or available. Cannot perform dynamic web crawling.")
        return []
    if not start_urls:
        logger.info("No starting URLs provided for crawling.")
        return []

    global MAX_PAGES_TO_CRAWL
    original_max_pages = MAX_PAGES_TO_CRAWL
    if max_pages_override is not None and max_pages_override > 0:
        logger.info(f"Overriding config MAX_PAGES_TO_CRAWL ({original_max_pages}) with {max_pages_override}")
        MAX_PAGES_TO_CRAWL = max_pages_override

    # Adjust concurrency based on whether PDFs are processed
    effective_max_concurrent = max_concurrent if process_pdfs_linked else max(3, max_concurrent // 2)
    logger.info(f"Starting Playwright web crawl from {len(start_urls)} URLs. Effective Max Concurrency: {effective_max_concurrent}. Max Pages/PDFs: {MAX_PAGES_TO_CRAWL}. Process Linked PDFs: {process_pdfs_linked}.")
    semaphore = asyncio.Semaphore(effective_max_concurrent)
    results = []

    # --- Playwright and aiohttp Session Management ---
    playwright = None
    browser = None
    context = None
    session = None
    try:
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=get_user_agent(),
            ignore_https_errors=True
        )
        # Create aiohttp session for direct PDF fetches
        connector = aiohttp.TCPConnector(limit_per_host=max(5, effective_max_concurrent // 2), limit=max(10, effective_max_concurrent * 2), ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            results = await _crawl_manager_pw( # Use Playwright manager
                start_urls=start_urls,
                playwright_context=context,
                session=session,
                semaphore=semaphore,
                process_pdfs=process_pdfs_linked,
                proxies=proxies
            )
            logger.info(f"Playwright web crawl finished. Returning {len(results)} processed items.")

    except Exception as e:
         logger.error(f"Playwright web crawl failed: {e}", exc_info=True)
    finally:
        # --- Cleanup ---
        if context: await context.close()
        if browser: await browser.close()
        if playwright: await playwright.stop()
        if max_pages_override is not None:
            MAX_PAGES_TO_CRAWL = original_max_pages
            logger.info(f"Restored MAX_PAGES_TO_CRAWL to {original_max_pages}")
        logger.info("Playwright resources cleaned up.")
        # ---
    return results


# --- arXiv Fetcher (Unchanged) ---
def fetch_arxiv_papers(query: str, max_results: int) -> List[Dict[str, Any]]:
    """
    Fetches paper metadata and abstracts from arXiv using its API client.
    """
    logger.info(f"Querying arXiv for '{query}' (max_results: {max_results})")
    if max_results <= 0: logger.warning("max_results must be positive for arXiv query."); return []
    client = arxiv.Client(page_size = min(max_results, 500), delay_seconds = 5, num_retries = 3)
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    results_list: List[Dict[str, Any]] = []
    fetched_count = 0
    try:
        results_generator = client.results(search)
        for result in results_generator:
            if fetched_count >= max_results: logger.info(f"Reached max_results ({max_results}) for arXiv query '{query}'. Stopping fetch."); break
            title = "Untitled"
            if result.title: title = ' '.join(result.title.splitlines()).strip()
            clean_summary = "No abstract available."
            if result.summary: clean_summary = ' '.join(result.summary.splitlines()).strip()
            content = f"{title}\n\n{clean_summary}" # Keep summary as content for arXiv
            authors = [str(author) for author in result.authors] if result.authors else []
            published_date = result.published.isoformat() if result.published else None
            entry_id = result.entry_id
            paper_url = result.pdf_url if result.pdf_url else entry_id
            if not paper_url: paper_url = entry_id; logger.warning(f"Could not find PDF or entry_id URL for arXiv result: {title}")
            paper_data = {"source": CONTENT_TYPE_ARXIV, "entry_id": entry_id, "title": title, "content": content, "authors": authors, "published": published_date, "url": paper_url}
            results_list.append(paper_data)
            fetched_count += 1
            logger.debug(f"Fetched arXiv paper {fetched_count}/{max_results}: {title}")
        logger.info(f"Successfully fetched {len(results_list)} papers from arXiv for query '{query}'.")
    except Exception as e:
        logger.error(f"An error occurred while fetching from arXiv for query '{query}': {e}", exc_info=True)
    return results_list


# --- Example Usage (Optional) ---
async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # --- Example: Crawl OpenReview ICML page and process PDFs ---
    start_web_urls = ["https://openreview.net/group?id=ICML.cc/2024/Conference#tab-accept-oral"]
    proxies_list = None
    # Set a low page limit for testing, but allow PDF processing
    print(f"\n--- Starting Playwright Web Crawl from: {start_web_urls} (Max Items: 50, Process PDFs: True) ---") # Increased limit for testing
    web_results = await crawl_and_fetch_web_articles(
        start_urls=start_web_urls,
        process_pdfs_linked=True, # Explicitly enable PDF processing
        max_pages_override=50,    # Limit total items (HTML pages + PDFs) fetched
        proxies=proxies_list,
        max_concurrent=10 # Increase concurrency
    )
    print(f"\n--- Web Crawl Results ({len(web_results)}) ---")
    for i, page in enumerate(web_results):
        print(f"{i+1}. Title: {page['title']}")
        print(f"   URL: {page['url']}")
        print(f"   Type: {page['source']}")
        if page.get('found_on'): print(f"   Found on: {page['found_on']}")
        print("-" * 15)


if __name__ == "__main__":
    # Ensure Playwright is installed before running
    if not PLAYWRIGHT_AVAILABLE:
         print("Playwright is required but not installed. Please run:", file=sys.stderr)
         print("  pip install playwright", file=sys.stderr)
         print("  playwright install", file=sys.stderr)
         sys.exit(1)
    asyncio.run(main())
