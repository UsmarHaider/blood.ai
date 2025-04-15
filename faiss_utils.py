# faiss_utils.py
import faiss
import pickle
import numpy as np
import os
import streamlit as st
import google.generativeai as genai
import logging
from config import FAISS_INDEX_PATH, FAISS_METADATA_PATH, EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION

logger = logging.getLogger('bloodcell_app')

# Initialize embedding model client (configure API key in app.py)
# No need to store the model object itself globally here

def generate_gemini_embedding(text: str, dimension: int = EMBEDDING_DIMENSION):
    """Generate an embedding for a text using Google's Gemini embedding model."""
    if not text or not isinstance(text, str):
        logger.warning("Attempted to generate embedding for empty or invalid text.")
        return None
    try:
        logger.debug(f"Generating embedding for text snippet (len={len(text)})...")
        # Configure embedding parameters
        embed_config = {}
        if dimension is not None and dimension > 0:
            embed_config["output_dimensionality"] = dimension

        # Generate embedding using the embedding model
        result = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=text,
            task_type="RETRIEVAL_DOCUMENT", # Use RETRIEVAL_DOCUMENT for indexing content
            **embed_config
        )
        logger.debug("Embedding generated successfully.")
        # Return the embedding values as a numpy array
        return np.array(result["embedding"], dtype=np.float32)
    except Exception as e:
        logger.error(f"Error generating Gemini embedding: {e}")
        st.error(f"Error generating embedding: {e}")
        return None

def generate_gemini_query_embedding(text: str, dimension: int = EMBEDDING_DIMENSION):
    """Generate an embedding for a query using Google's Gemini embedding model."""
    if not text or not isinstance(text, str):
        logger.warning("Attempted to generate query embedding for empty or invalid text.")
        return None
    try:
        logger.debug(f"Generating query embedding for: {text[:50]}...")
        embed_config = {}
        if dimension is not None and dimension > 0:
            embed_config["output_dimensionality"] = dimension

        result = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=text,
            task_type="RETRIEVAL_QUERY", # Use RETRIEVAL_QUERY for search queries
            **embed_config
        )
        logger.debug("Query embedding generated.")
        return np.array(result["embedding"], dtype=np.float32)
    except Exception as e:
        logger.error(f"Error generating Gemini query embedding: {e}")
        st.error(f"Error generating query embedding: {e}")
        return None


def load_faiss_index():
    """Load the FAISS index and metadata if they exist."""
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_METADATA_PATH):
        try:
            logger.info(f"Loading FAISS index from {FAISS_INDEX_PATH}")
            index = faiss.read_index(FAISS_INDEX_PATH)
            logger.info(f"Loading FAISS metadata from {FAISS_METADATA_PATH}")
            with open(FAISS_METADATA_PATH, 'rb') as f:
                metadata = pickle.load(f)
            logger.info(f"FAISS index ({index.ntotal} vectors) and metadata loaded.")
            return index, metadata
        except Exception as e:
            logger.warning(f"Failed to load existing FAISS index/metadata: {e}. Will create a new one.")
            # Clean up potentially corrupted files
            if os.path.exists(FAISS_INDEX_PATH): os.remove(FAISS_INDEX_PATH)
            if os.path.exists(FAISS_METADATA_PATH): os.remove(FAISS_METADATA_PATH)
            return None, None
    logger.info("No existing FAISS index found. A new one will be created.")
    return None, None

def create_new_faiss_index(dimension: int = EMBEDDING_DIMENSION):
    """Creates a new FAISS index and empty metadata dictionary."""
    logger.info(f"Creating new FAISS index with dimension {dimension}.")
    index = faiss.IndexFlatL2(dimension)
    metadata = {
        'texts': [],      # Original text chunks
        'urls': [],       # Source URLs for each chunk
        'timestamps': []  # When the chunk was added
    }
    return index, metadata

def search_faiss_index(index, metadata, query_text, k=3, threshold=20.0):
    """Searches the FAISS index for text relevant to the query."""
    relevant_texts = []
    if index is None or index.ntotal == 0:
        logger.info("FAISS index is empty. No search performed.")
        return relevant_texts # Return empty list if index is empty

    try:
        # Generate embedding for the user query using Gemini
        logger.debug(f"Searching FAISS for query: {query_text[:50]}...")
        query_embedding = generate_gemini_query_embedding(query_text, dimension=index.d) # Use index dimension

        if query_embedding is not None:
            # Reshape to 2D array for faiss search
            query_embedding = query_embedding.reshape(1, -1)

            # Search the index
            actual_k = min(k, index.ntotal) # Ensure k is not larger than index size
            distances, indices = index.search(query_embedding, actual_k)

            # Get the relevant texts based on distance threshold
            for i in range(actual_k):
                idx = indices[0][i]
                dist = distances[0][i]
                if dist < threshold: # Only include if distance is reasonable (lower is better for L2)
                    if idx < len(metadata['texts']):
                         relevant_texts.append(metadata['texts'][idx])
                         logger.debug(f"Found relevant text (idx={idx}, dist={dist:.2f})")
                    else:
                        logger.warning(f"FAISS index returned idx {idx} which is out of bounds for metadata texts (len={len(metadata['texts'])}). Metadata might be corrupted.")
                else:
                    logger.debug(f"Text (idx={idx}) skipped due to distance {dist:.2f} > threshold {threshold}")


            logger.info(f"FAISS search complete. Found {len(relevant_texts)} relevant text(s).")
        else:
             logger.warning("Could not generate query embedding. FAISS search skipped.")

    except Exception as e:
        logger.error(f"Error searching FAISS index: {e}")
        st.warning(f"Error searching knowledge base: {e}")

    return relevant_texts