# hybrid_search_rag/data/resource_fetcher.py
"""Fetches data from arXiv and web sources (HTML/PDF)."""

import os
import re
import arxiv
import fitz # PyMuPDF
import requests
from bs4 import BeautifulSoup
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import time
import io
from urllib.parse import urljoin, urlparse
from collections import deque

logger = logging.getLogger(__name__)

# --- Request Settings ---
FETCH_TIMEOUT = 60 # seconds
HEAD_TIMEOUT = 10 # seconds

# --- Crawling Settings ---
from .. import config
try:
    MAX_PAGES_TO_CRAWL = config.MAX_PAGES_TO_CRAWL
    logger.info(f"Crawler using MAX_PAGES_TO_CRAWL = {MAX_PAGES_TO_CRAWL} (from config).")
except AttributeError:
    logger.error("MAX_PAGES_TO_CRAWL not found in config.py! Using default: 50.")
    MAX_PAGES_TO_CRAWL = 50
CRAWL_DELAY_SECONDS = config.CRAWL_DELAY_SECONDS
ALLOWED_DOMAINS = [] # Default: same-domain crawling only.

# --- Constants for Content Types ---
CONTENT_TYPE_HTML = "html"
CONTENT_TYPE_PDF = "pdf"
CONTENT_TYPE_OTHER = "other"
CONTENT_TYPE_FAILED = "failed"

def _clean_text(text: str) -> str:
    """Cleans extracted text by normalizing whitespace and reducing excessive newlines."""
    if not text: return ""
    try:
        text = re.sub(r'[ \t]+', ' ', text) # Normalize spaces/tabs
        lines = text.splitlines()
        cleaned_lines = [line.strip() for line in lines if line.strip()] # Strip lines and remove empty ones
        if not cleaned_lines: return ""
        text = "\n".join(cleaned_lines)
        text = re.sub(r'\n{3,}', '\n\n', text) # Reduce 3+ newlines to 2 (paragraph breaks)
        return text.strip()
    except Exception as e:
        logger.error(f"Text cleaning failed: {e}")
        return text or ""

# --- Refactored Core Fetcher ---
def _fetch_and_parse_content(url: str) -> Tuple[str, str, Optional[str], Optional[BeautifulSoup]]:
    """
    Fetches URL, determines type, extracts text, and returns parsed content.
    Returns: (final_url, content_type_constant, text_content, soup_object_or_None)
    """
    logger.info(f"Fetching: {url}")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    final_url = url

    try:
        # HEAD Request to check Content-Type
        content_type_header = ''
        try:
            head_response = requests.head(url, headers=headers, timeout=HEAD_TIMEOUT, allow_redirects=True)
            head_response.raise_for_status()
            content_type_header = head_response.headers.get('Content-Type', '').lower()
            final_url = head_response.url
        except requests.exceptions.RequestException as e:
            logger.warning(f"HEAD request failed for {url}: {e}. Aborting fetch.")
            return url, CONTENT_TYPE_FAILED, None, None

        # Determine Type
        is_pdf = 'application/pdf' in content_type_header
        is_html = 'text/html' in content_type_header

        # GET Request for full content
        response = requests.get(final_url, headers=headers, timeout=FETCH_TIMEOUT)
        response.raise_for_status()
        final_url = response.url # Update again

        # Process Based on Type
        if is_pdf:
            pdf_text = ""
            try:
                with fitz.open(stream=response.content, filetype="pdf") as doc:
                    # logger.info(f"Extracting text from PDF ({doc.page_count} pages): {final_url}") # Maybe too verbose
                    for i, page in enumerate(doc):
                        try:
                            pdf_text += page.get_text("text", sort=True) + "\n\n"
                        except Exception as page_err:
                            logger.warning(f"Failed to extract text from page {i+1} of PDF {final_url}: {page_err}")
                cleaned_text = _clean_text(pdf_text)
                if cleaned_text:
                    return final_url, CONTENT_TYPE_PDF, cleaned_text, None
                else:
                    logger.warning(f"No text extracted from PDF: {final_url}")
                    return final_url, CONTENT_TYPE_PDF, None, None
            except Exception as e:
                logger.error(f"PDF processing failed for {final_url}: {e}", exc_info=True)
                return final_url, CONTENT_TYPE_PDF, None, None

        elif is_html:
            try:
                response.encoding = response.apparent_encoding
                soup = BeautifulSoup(response.text, 'html.parser')
                main_content_area = soup.find('article') or soup.find('main') or soup.body
                cleaned_text = None
                if main_content_area:
                    raw_text = main_content_area.get_text(separator='\n', strip=True)
                    cleaned_text = _clean_text(raw_text)
                else:
                    logger.warning(f"No main content tags (<article>, <main>) found in HTML: {final_url}")
                # Return soup object for HTML to allow title/link extraction later
                return final_url, CONTENT_TYPE_HTML, cleaned_text, soup
            except Exception as e:
                logger.error(f"HTML parsing failed for {final_url}: {e}", exc_info=True)
                return final_url, CONTENT_TYPE_HTML, None, None

        else: # Other content type
            logger.info(f"Skipping content extraction for non-PDF/HTML type '{content_type_header}' at {final_url}")
            return final_url, CONTENT_TYPE_OTHER, None, None

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP Request failed for URL {url} (final URL tried: {final_url}): {e}")
        return final_url if 'final_url' in locals() else url, CONTENT_TYPE_FAILED, None, None
    except Exception as e:
        logger.error(f"Unexpected error processing URL {url}: {e}", exc_info=True)
        return url, CONTENT_TYPE_FAILED, None, None

# --- Helper for Domain Checking ---
def get_domain(url: str) -> Optional[str]:
    """Extracts the 'netloc' (e.g., 'google.com') from a URL."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return None

# --- Refactored Web Crawler ---
def crawl_and_fetch_web_articles(start_urls: List[str]) -> List[Dict[str, Any]]:
    """
    Web crawler: Explores URLs, fetches content, extracts titles, handles PDFs.
    Respects limits and domain filtering. Stops queueing new links when target hit.
    """
    if not start_urls:
        logger.info("No starting URLs provided for crawling.")
        return []

    results_list: List[Dict[str, Any]] = []
    queue = deque(start_urls)
    visited = set(start_urls)
    start_domains = {get_domain(url) for url in start_urls if get_domain(url)}
    visited_set_limit = MAX_PAGES_TO_CRAWL * 5
    stop_queuing_new_links = False

    logger.info(f"Starting crawl from {len(start_urls)} URLs. Target pages: {MAX_PAGES_TO_CRAWL}. Delay: {CRAWL_DELAY_SECONDS}s.")

    while queue and len(results_list) < MAX_PAGES_TO_CRAWL:
        current_url = queue.popleft()
        # logger.info(f"Processing URL from queue ({len(queue)} left): {current_url}") # Can be verbose
        time.sleep(CRAWL_DELAY_SECONDS)

        try:
            # Fetch and parse content using the refactored function
            final_url, content_type, text_content, soup = _fetch_and_parse_content(current_url)

            # Mark redirected URL as visited
            if final_url != current_url and final_url not in visited:
                if len(visited) < visited_set_limit:
                    visited.add(final_url)
                # else: Limit hit, flag handles queuing later

            # Process if fetch succeeded and content was extracted
            if content_type != CONTENT_TYPE_FAILED and text_content:
                if len(results_list) < MAX_PAGES_TO_CRAWL:
                    # Get Title
                    page_title = final_url # Default
                    if content_type == CONTENT_TYPE_HTML and soup:
                        try:
                            title_tag = soup.find('title')
                            if title_tag and title_tag.string:
                                page_title = title_tag.string.strip()
                        except Exception as title_err:
                             logger.warning(f"Error extracting title for {final_url}: {title_err}.")
                    elif content_type == CONTENT_TYPE_PDF:
                        try: # Try getting filename for PDF title
                            pdf_filename = os.path.basename(urlparse(final_url).path)
                            if pdf_filename: page_title = pdf_filename
                        except Exception: pass

                    # Store Result Data
                    result_data = {
                        "source": content_type, "url": final_url, "title": page_title,
                        "content": f"{page_title}. {text_content}",
                        "authors": [], "published": None,
                    }
                    results_list.append(result_data)
                    # Log progress less frequently or only title snippet?
                    # logger.info(f"Processed: {final_url} ({len(results_list)}/{MAX_PAGES_TO_CRAWL}) Type: {content_type}, Title: {page_title[:60]}...")

                    # Check if target hit, stop queuing new links
                    if not stop_queuing_new_links and len(results_list) >= MAX_PAGES_TO_CRAWL:
                        logger.warning(f"Target pages ({MAX_PAGES_TO_CRAWL}) reached. Stopping queuing new links.")
                        stop_queuing_new_links = True
                else:
                    # Already hit the limit before processing this page's content.
                    pass # No need to log skipping storage every time

                # Process Discovered Links (Only if HTML and not stopping)
                if content_type == CONTENT_TYPE_HTML and soup and not stop_queuing_new_links:
                    discovered_links = []
                    base_url = final_url
                    main_content_area = soup.find('article') or soup.find('main') or soup.body
                    if main_content_area:
                        for link_tag in main_content_area.find_all('a', href=True):
                            href = link_tag['href'].strip()
                            if href and not href.startswith(('#', 'mailto:', 'javascript:')):
                                absolute_url = urljoin(base_url, href)
                                discovered_links.append(absolute_url)
                    # logger.debug(f"Found {len(discovered_links)} potential links in HTML for {final_url}") # Debug level

                    # Queue Valid Links
                    current_domain = get_domain(final_url)
                    links_queued_this_page = 0
                    for link_url in discovered_links:
                        # Standard Filters
                        if link_url in visited: continue
                        if not link_url.startswith(('http://', 'https://')): continue
                        parsed_link_path = urlparse(link_url).path.lower()
                        # Removed PDF filter, kept others
                        skipped_extensions = ['.jpg','.jpeg','.png','.gif','.bmp','.svg','.zip','.rar','.tar.gz',
                                              '.css','.js','.xml','.json','.txt','.doc','.docx','.xls','.xlsx','.ppt','.pptx','.mp3','.mp4','.avi']
                        if any(parsed_link_path.endswith(ext) for ext in skipped_extensions): continue
                        link_domain = get_domain(link_url)
                        if not link_domain: continue
                        if ALLOWED_DOMAINS: # Check allowed list if provided
                            if link_domain not in ALLOWED_DOMAINS: continue
                        elif start_domains: # Otherwise check start domains
                            if link_domain not in start_domains: continue
                        elif not start_domains: continue # Skip if no domain rules defined

                        # Add to queue if limits not hit
                        if len(visited) < visited_set_limit:
                            visited.add(link_url)
                            queue.append(link_url)
                            links_queued_this_page += 1
                        else: # Hit visited safety limit
                            if not stop_queuing_new_links:
                                logger.warning(f"Visited set safety limit ({visited_set_limit}) reached. Halting link discovery.")
                                stop_queuing_new_links = True
                            break # Stop processing links for THIS page
                    # if links_queued_this_page > 0: # Maybe too verbose
                    #    logger.debug(f"Added {links_queued_this_page} new valid links to queue from {final_url}.")

            elif content_type == CONTENT_TYPE_FAILED:
                # Already logged warning/error in fetcher function
                pass
            # else: Type was OTHER or no text content, already logged in fetcher

        except Exception as e:
            logger.error(f"Crawling/processing {current_url} failed unexpectedly: {e}", exc_info=True)

    # --- End of Main Crawl Loop ---
    logger.info(f"Crawling finished. Processed {len(results_list)} pages.")
    if len(queue) > 0 and len(results_list) >= MAX_PAGES_TO_CRAWL:
        logger.info(f"Note: Queue still contained {len(queue)} URLs when target page count met.")
    elif len(queue) == 0:
        logger.info("Processing queue is empty.")
    return results_list

# --- arXiv Fetcher ---
def fetch_arxiv_papers(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Fetches paper metadata and abstracts from arXiv using its API."""
    logger.info(f"Querying arXiv for '{query}' (max: {max_results})")
    if max_results <= 0: return []

    client = arxiv.Client(page_size = min(max_results, 1000), delay_seconds = 3, num_retries = 3)
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    results_list: List[Dict[str, Any]] = []
    fetched_count = 0
    try:
        for result in client.results(search):
            if fetched_count >= max_results: break
            clean_summary = (result.summary or "").replace('\n', ' ').strip()
            title = (result.title or "Untitled").strip()
            content = f"{title}. {clean_summary}"
            paper_data = {
                "source": "arxiv", "entry_id": result.entry_id, "title": title,
                "content": content, "authors": [str(author) for author in result.authors],
                "published": result.published.isoformat() if result.published else None,
                "url": result.pdf_url or result.entry_id
            }
            results_list.append(paper_data)
            fetched_count += 1
        logger.info(f"Fetched {len(results_list)} papers from arXiv for query '{query}'.")
    except Exception as e:
        logger.error(f"An error occurred while fetching from arXiv: {e}", exc_info=True)
    finally:
        return results_list