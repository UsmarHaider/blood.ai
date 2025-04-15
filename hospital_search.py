# hospital_search.py
import streamlit as st
import time
import logging
import sys
import os

# --- Configuration (Import relevant constants) ---
from config import HOSPITAL_CACHE_DURATION

logger = logging.getLogger('bloodcell_app')

# --- AgentPro Import Attempt ---
# Define placeholders first in case import fails
class AgentProPlaceholder:
    def __init__(self, *args, **kwargs):
        logger.warning("AgentProPlaceholder initialized. AgentPro library not found or failed to load.")
        pass
    def __call__(self, *args, **kwargs):
        return "AgentPro is not available. Hospital search functionality requires the AgentPro library."

class AresInternetToolPlaceholder:
    def __init__(self):
        self.name = "AresInternetTool"
        self.description = "A tool for searching the internet (currently unavailable)"

AgentPro = AgentProPlaceholder
AresInternetTool = AresInternetToolPlaceholder
agent = None
hospital_search_available = False

try:
    # Try standard import first
    from AgentPro.agentpro import AgentPro as ActualAgentPro
    from AgentPro.agentpro.tools import AresInternetTool as ActualAresTool
    AgentPro = ActualAgentPro # Overwrite placeholder if successful
    AresInternetTool = ActualAresTool
    logger.info("Attempting to initialize AgentPro...")
    tools = [AresInternetTool()]
    agent = AgentPro(tools=tools)
    logger.info("AgentPro initialized successfully with AresInternetTool.")
    hospital_search_available = True
except ImportError:
    logger.warning("Standard AgentPro import failed.")
    # Try adding parent directory if AgentPro is local
    try:
        agentpro_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AgentPro")
        if os.path.isdir(agentpro_path):
             sys.path.append(os.path.dirname(agentpro_path)) # Add parent of AgentPro dir
             logger.info(f"Added {os.path.dirname(agentpro_path)} to sys.path for local AgentPro.")
             from agentpro import AgentPro as LocalAgentPro
             from agentpro.tools import AresInternetTool as LocalAresTool
             AgentPro = LocalAgentPro # Overwrite placeholder
             AresInternetTool = LocalAresTool
             logger.info("Attempting to initialize local AgentPro...")
             tools = [AresInternetTool()]
             agent = AgentPro(tools=tools)
             logger.info("Local AgentPro initialized successfully.")
             hospital_search_available = True
        else:
             logger.warning("Local AgentPro directory not found. Hospital search disabled.")
             # Keep placeholders active
    except ImportError as e:
        logger.error(f"Failed to import AgentPro even after path modification: {e}. Hospital search disabled.")
        # Keep placeholders active
    except Exception as e_init:
        logger.error(f"Error initializing AgentPro (even if imported): {e_init}")
        agent = None # Ensure agent is None if init fails
        hospital_search_available = False
except Exception as e_general:
    logger.error(f"An unexpected error occurred during AgentPro setup: {e_general}")
    agent = None # Ensure agent is None
    hospital_search_available = False


# --- Hospital Search Functionality ---
HOSPITAL_CACHE = {}  # In-memory cache: {(disease, location): (timestamp, results)}

def search_hospitals(disease: str, location: str, force_refresh: bool = False) -> str:
    """
    Search for hospitals specializing in treating a specific blood disease/indicator
    in a given location using the configured agent.
    """
    global agent # Make sure we are using the agent initialized above

    if not hospital_search_available or agent is None:
        logger.warning("Hospital search requested, but AgentPro is not available.")
        return "⚠️ Hospital search functionality is currently unavailable. Please ensure the AgentPro library is installed and configured correctly."

    cache_key = (disease, location)

    # Check cache first (if not forcing refresh)
    if not force_refresh and cache_key in HOSPITAL_CACHE:
        timestamp, results = HOSPITAL_CACHE[cache_key]
        if time.time() - timestamp < HOSPITAL_CACHE_DURATION:
            logger.info(f"Using cached hospital results for {disease} in {location}")
            return results
        else:
            logger.info(f"Cache expired for {disease} in {location}. Performing fresh search.")

    try:
        # Format the query for better results based on the type of 'disease'
        if disease.upper() in ["NPM1", "PML_RARA", "RUNX1_RUNX1T1", "CONTROL"]: # Handle genetic markers/control
            query = (
                f"Find hospitals or specialized hematology/oncology centers in {location}, Pakistan "
                f"that are equipped to diagnose and manage patients with blood disorders, specifically mentioning expertise "
                f"related to genetic markers like '{disease}' if applicable. List the top 3 relevant facilities. "
                f"For each, provide: Name, Address, Contact Number (if available), and known specialization/expertise in hematological malignancies. "
                f"Format the response clearly using markdown headings for each hospital and bullet points for details."
            )
        else: # General query for other conditions/cell types (though searching hospitals for cell types might be less direct)
             query = (
                f"Find hospitals or medical centers in {location}, Pakistan known for strong hematology or oncology departments "
                f"that treat blood-related conditions like those indicated by findings related to '{disease}'. List the top 3. "
                 f"For each, provide: Name, Address, Contact Number (if available), and general expertise in blood disorders. "
                 f"Format the response clearly using markdown headings for each hospital and bullet points for details."
            )

        logger.info(f"Executing hospital search query via AgentPro: {query}")

        # Execute the search using the initialized AgentPro agent
        response = agent(query) # AgentPro call

        # Process and format the response
        formatted_response = _format_hospital_response(response, disease, location)

        # Cache the result
        HOSPITAL_CACHE[cache_key] = (time.time(), formatted_response)
        logger.info(f"Cached new hospital results for {disease} in {location}")

        return formatted_response

    except Exception as e:
        error_msg = f"Error during hospital search via AgentPro: {str(e)}"
        logger.error(error_msg)
        # Provide specific feedback if it's likely an AgentPro issue
        if "AgentPro" in str(e) or "AresInternetTool" in str(e):
             return f"⚠️ An error occurred with the hospital search agent: {str(e)}\nPlease check the agent configuration or try again later."
        else:
             return f"⚠️ An unexpected error occurred during the hospital search: {str(e)}\nPlease try again later."


def _format_hospital_response(response: str, disease: str, location: str) -> str:
    """Format the hospital search response for better readability."""
    if not response or not isinstance(response, str) or len(response.strip()) < 20 or "not available" in response.lower() or "could not find" in response.lower():
        logger.warning(f"Received empty or non-informative response from AgentPro for {disease} in {location}.")
        # Try a slightly more helpful message depending on the input
        search_term = f"'{disease}' genetic marker analysis/treatment" if disease.upper() in ["NPM1", "PML_RARA", "RUNX1_RUNX1T1"] else f"'{disease}' condition"
        return (f"ℹ️ The search agent could not find specific hospitals listed for **{search_term}** in **{location}**.\n\n"
                f"**Recommendation:** Please consult a general hematologist or oncologist in {location}. They can provide referrals to specialized centers if needed. Major university hospitals or cancer centers are often well-equipped.")

    # Add a header and disclaimer
    formatted_response = f"""
### Hospital Information for {disease} in {location}

Based on the search, here are some potentially relevant facilities:

{response.strip()}

---

**Disclaimer:** This information is generated by an AI search agent and may not be exhaustive or perfectly accurate. Details like specialization, availability of specific tests/treatments, and contact information **must be verified directly** with the hospitals. This is **not** a medical recommendation. Always consult with a qualified healthcare provider for medical advice and referrals.
"""
    logger.info("Formatted hospital search response successfully.")
    return formatted_response