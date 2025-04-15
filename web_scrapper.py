# web_scraper.py
import requests
from bs4 import BeautifulSoup
import streamlit as st
import os
import time
import pickle
import faiss
import logging

# Import necessary functions and objects from other modules
from config import DATA_FILE_PATH, FAISS_INDEX_PATH, FAISS_METADATA_PATH, EMBEDDING_DIMENSION
from faiss_utils import generate_gemini_embedding # Use the function from faiss_utils

logger = logging.getLogger('bloodcell_app')

def scrape_website_content(url: str) -> tuple[bool, str]:
    """
    Attempts to scrape paragraphs and headings from a given URL.

    Returns:
        tuple[bool, str]: (success_status, content_or_error_message)
    """
    logger.info(f"Attempting to scrape content from URL: {url}")
    try:
        # Add scheme if missing
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            logger.debug(f"Added scheme: {url}")

        # Basic headers to mimic a browser
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

        # Make the request
        response = requests.get(url, headers=headers, timeout=15) # Increased timeout
        response.raise_for_status() # Check for HTTP errors (4xx, 5xx)
        logger.info(f"Successfully fetched URL {url} with status code {response.status_code}.")

        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Attempt to find the main content area (common patterns)
        main_content = soup.find('main') or \
                       soup.find('article') or \
                       soup.find('div', id='content') or \
                       soup.find('div', class_='content') or \
                       soup.find('div', id='main') or \
                       soup.find('div', class_='main')

        if main_content:
             logger.debug("Found potential main content container.")
             target_soup = main_content # Search within main content
        else:
             logger.debug("No specific main content container found, searching entire body.")
             target_soup = soup.body # Fallback to searching the whole body

        if not target_soup:
             logger.warning(f"Could not find body tag in the response for {url}.")
             return False, f"Error processing {url}: Could not find body tag."


        # Extract relevant text elements (paragraphs, headings, list items)
        elements = target_soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li'])

        content_parts = []
        min_length = 15 # Ignore very short tags

        for element in elements:
             # Clean whitespace and check length
             text = element.get_text(separator=' ', strip=True)
             if text and len(text) >= min_length:
                 # Add appropriate prefix for headings for readability
                 if element.name.startswith('h'):
                     content_parts.append(f"\n## {text}\n") # Markdown heading style
                 else:
                     content_parts.append(text)


        # Combine and perform final cleaning
        content = '\n\n'.join(content_parts)
        content = '\n'.join(line.strip() for line in content.splitlines() if line.strip()) # Remove empty lines

        if not content:
            logger.warning(f"Could not find significant text content (p, h, li tags) in {url}.")
            return False, f"Could not find significant text content on {url} using common tags (p, h1-4, li)."

        logger.info(f"Successfully scraped {len(content)} characters from {url}.")
        return True, content

    except requests.exceptions.Timeout:
        logger.error(f"Timeout error fetching URL {url}")
        return False, f"Error fetching URL {url}: The request timed out."
    except requests.exceptions.HTTPError as e:
         logger.error(f"HTTP error fetching URL {url}: {e}")
         return False, f"Error fetching URL {url}: HTTP {e.response.status_code} - {e.response.reason}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Network/Request error fetching URL {url}: {str(e)}")
        return False, f"Error fetching URL {url}: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error processing {url}: {str(e)}")
        return False, f"Unexpected error processing {url}: {str(e)}"


def save_content_to_knowledge_base(url: str, content: str, faiss_index, metadata):
    """
    Save the scraped content to data.txt and update FAISS index & metadata.

    Args:
        url (str): The source URL.
        content (str): The scraped text content.
        faiss_index: The FAISS index object.
        metadata (dict): The metadata dictionary associated with the index.

    Returns:
        tuple[bool, str]: (success_status, message)
    """
    if not content or not isinstance(content, str):
        return False, "No valid content provided to save."

    # 1. Save to data.txt (append mode)
    try:
        with open(DATA_FILE_PATH, 'a', encoding='utf-8') as file:
            file.write(f"\n\n--- CONTENT FROM: {url} ---\n")
            file.write(content)
            file.write("\n--- END CONTENT ---\n")
        logger.info(f"Appended content from {url} to {DATA_FILE_PATH}.")
    except Exception as e:
        logger.error(f"Error saving content to {DATA_FILE_PATH}: {e}")
        return False, f"Error saving to data file: {e}"

    # 2. Create embeddings and add to FAISS index
    if faiss_index is None or metadata is None:
        logger.error("FAISS index or metadata is None. Cannot add embeddings.")
        return False, "FAISS index not initialized. Cannot save embeddings."

    try:
        # Split content into manageable chunks (e.g., by paragraphs or fixed size)
        # Using paragraphs first, then splitting large paragraphs if needed
        max_chunk_chars = 800 # Adjust based on embedding model limits and desired granularity
        chunks = []
        current_chunk = ""

        for paragraph in content.split('\n\n'): # Split by double newline first
             paragraph = paragraph.strip()
             if not paragraph:
                 continue

             if len(current_chunk) + len(paragraph) + 2 < max_chunk_chars: # +2 for potential \n\n
                 if current_chunk:
                     current_chunk += "\n\n" + paragraph
                 else:
                     current_chunk = paragraph
             else:
                 # If current chunk has content, add it
                 if current_chunk:
                     chunks.append(current_chunk)
                 # Handle the new paragraph - it might be too long itself
                 if len(paragraph) < max_chunk_chars:
                     current_chunk = paragraph
                 else:
                     # Split the long paragraph further (e.g., by sentences or fixed length)
                     # Simple fixed length splitting for now:
                     for i in range(0, len(paragraph), max_chunk_chars):
                         chunks.append(paragraph[i:i + max_chunk_chars])
                     current_chunk = "" # Reset current chunk after splitting the long one

        if current_chunk: # Add the last chunk
             chunks.append(current_chunk)

        logger.info(f"Split content from {url} into {len(chunks)} chunks for embedding.")

        # Create embeddings for each chunk
        embedding_success_count = 0
        new_embeddings = []
        new_metadata_indices = []

        with st.spinner(f"Generating {len(chunks)} embeddings for '{url}'..."):
            for i, chunk in enumerate(chunks):
                 logger.debug(f"Generating embedding for chunk {i+1}/{len(chunks)} (len={len(chunk)})...")
                 embedding = generate_gemini_embedding(chunk, dimension=EMBEDDING_DIMENSION)

                 if embedding is not None:
                     new_embeddings.append(embedding)
                     # Store index relative to the start of *this* batch of additions
                     new_metadata_indices.append(len(metadata['texts']))
                     # Append metadata immediately
                     metadata['texts'].append(chunk)
                     metadata['urls'].append(url)
                     metadata['timestamps'].append(time.time())
                     embedding_success_count += 1
                 else:
                     logger.warning(f"Failed to generate embedding for chunk {i+1} from {url}.")
                     st.warning(f"Could not generate embedding for a chunk of text from {url}. Skipping.")


        # 3. Add embeddings to FAISS index in batch if any were successful
        if new_embeddings:
             embeddings_array = np.array(new_embeddings).astype('float32').reshape(-1, EMBEDDING_DIMENSION)
             faiss_index.add(embeddings_array)
             logger.info(f"Added {embedding_success_count} new embeddings to FAISS index.")

             # 4. Save the updated index and metadata
             try:
                 faiss.write_index(faiss_index, FAISS_INDEX_PATH)
                 logger.info(f"Saved updated FAISS index to {FAISS_INDEX_PATH}")
                 with open(FAISS_METADATA_PATH, 'wb') as f:
                     pickle.dump(metadata, f)
                 logger.info(f"Saved updated FAISS metadata to {FAISS_METADATA_PATH}")
                 return True, f"Successfully extracted content, saved to file, and added {embedding_success_count}/{len(chunks)} text chunks to the searchable knowledge base."
             except Exception as e_save:
                 logger.error(f"Error saving updated FAISS index/metadata: {e_save}")
                 # Attempt to revert metadata changes if save failed? Complex, maybe just warn.
                 return False, f"Content saved to file, but failed to save updated FAISS index/metadata: {e_save}"
        elif embedding_success_count == 0 and len(chunks) > 0:
             return False, f"Content saved to file, but failed to generate any embeddings for the {len(chunks)} text chunks."
        else: # No chunks found or no successful embeddings
             return True, "Content saved to file, but no text chunks were processed for the knowledge base."


    except Exception as e:
        logger.error(f"Error creating or adding embeddings for {url}: {e}")
        # Return True because content was saved to file, but indicate embedding failure.
        return True, f"Content saved to file, but an error occurred during embedding creation: {e}"