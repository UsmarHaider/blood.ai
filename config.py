# config.py
import os
import streamlit as st
import logging

# --- Basic Setup ---
APP_TITLE = "AI Chat & Blood Classifier"
PAGE_LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "auto"

# --- API Keys & Model Names ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("❌ Google API Key not found in Streamlit secrets. Please add GOOGLE_API_KEY='YourKey' to .streamlit/secrets.toml")
    st.stop()
except Exception as e:
    st.error(f"❌ Error accessing Streamlit secrets: {e}")
    st.stop()

GEMINI_MODEL_NAME = "gemini-1.5-flash"  # Or "gemini-pro", etc.
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

# --- File Paths ---
# Get the directory where this config file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BLOOD_DISEASE_MODEL_PATH = os.path.join(BASE_DIR, 'blood_cells_model.h5')
CELL_TYPE_MODEL_PATH = os.path.join(BASE_DIR, 'image_classification_model.h5')
DATA_FILE_PATH = os.path.join(BASE_DIR, 'data.txt')
FAISS_INDEX_PATH = os.path.join(BASE_DIR, 'faiss_index.bin')
FAISS_METADATA_PATH = os.path.join(BASE_DIR, 'faiss_metadata.pkl')

# --- Model & Classification Settings ---
# Class names for the "Blood Disease" model
DISEASE_INDICATOR_CLASS_NAMES = ["RUNX1_RUNX1T1", "control", "NPM1", "PML_RARA"] # Adjusted based on common errors - check your model's actual classes
# Class names for the "Blood Cell Type" model
CELL_TYPE_CLASS_NAMES = ["ig", "lymphocyte", "monocyte", "neutrophil", "platelet"]

# Target image sizes for models (Check these against your model training)
BLOOD_DISEASE_TARGET_SIZE = (224, 224)
CELL_TYPE_TARGET_SIZE = (64, 64)

# --- FAISS Settings ---
EMBEDDING_DIMENSION = 768  # Default for text-embedding-004, adjust if needed

# --- Hospital Search Settings ---
HOSPITAL_CACHE_DURATION = 86400  # 24 hours in seconds

# --- Logging Setup ---
LOGGING_LEVEL = logging.INFO
LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def setup_logging():
    """Configures the application's logger."""
    logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)
    logger = logging.getLogger('bloodcell_app')
    return logger

# --- Available Locations ---
AVAILABLE_LOCATIONS = ["Lahore", "Karachi", "Islamabad", "Multan", "Faisalabad", "Peshawar"]