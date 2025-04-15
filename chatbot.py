# chatbot.py
import streamlit as st
import google.generativeai as genai
import logging
from config import GEMINI_MODEL_NAME, DATA_FILE_PATH
from faiss_utils import search_faiss_index # Import the search function

logger = logging.getLogger('bloodcell_app')

def load_knowledge_base():
    """Load the knowledge base from the data.txt file."""
    try:
        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as file:
            logger.info(f"Loading knowledge base from {DATA_FILE_PATH}")
            return file.read()
    except FileNotFoundError:
        logger.warning(f"Knowledge base file not found: {DATA_FILE_PATH}. Returning empty context.")
        return "" # Return empty string if not found, avoid error later

def get_system_prompt():
    """Creates the system prompt for the Gemini model, including loaded knowledge."""
    data_details = load_knowledge_base()

    # Add a check for empty data_details
    knowledge_base_section = ""
    if data_details and data_details.strip():
        knowledge_base_section = f"""Use the following specific details from the knowledge base if relevant to the user's question and within your scope:
---
{data_details}
---
"""
    else:
        knowledge_base_section = "No additional context data found in the knowledge base.\n"


    SYSTEM_PROMPT = f"""You are an AI assistant specialized in providing information about blood cell types and blood diseases. Your purpose is to offer general knowledge and explanations based on established medical information.

You can discuss:
* Different types of blood cells (e.g., lymphocytes, monocytes, neutrophils, platelets, red blood cells) and their functions.
* General information about common blood disorders, conditions, or indicators (e.g., anemia, leukemia, sickle cell disease, or specific genetic markers like NPM1 or PML-RARA mentioned in context).
* Basic concepts related to blood types (e.g., ABO system, Rh factor - but not determine a user's type).
* Definitions of related medical terms.

{knowledge_base_section}
**IMPORTANT LIMITATIONS: You MUST strictly adhere to the following:**
* **DO NOT provide medical diagnoses.** You cannot tell a user if they have a specific disease.
* **DO NOT interpret personal medical data,** such as lab results or medical images.
* **DO NOT offer medical advice,** treatment recommendations, or suggestions on managing health conditions.
* **DO NOT act as a substitute for a qualified healthcare professional.** Your information is for general knowledge only.

**If a user asks for a diagnosis, medical advice, interpretation of their personal results/images, or asks 'what disease do I have?', you MUST politely refuse.** State clearly that you are an informational AI assistant and cannot provide medical services. **Strongly advise the user to consult with a doctor or qualified healthcare provider** for any personal health concerns, diagnosis, or treatment.

Keep your responses informative, factual, objective, and strictly within the boundaries of providing general educational information. Avoid speculation.
"""
    return SYSTEM_PROMPT


def generate_chatbot_response(prompt: str, faiss_index, metadata):
    """Generates a response using the Gemini model, incorporating FAISS context."""
    assistant_response = "Sorry, something went wrong." # Default error message

    # 1. Search FAISS index for relevant context
    context_from_faiss = ""
    if faiss_index is not None and metadata is not None and faiss_index.ntotal > 0:
        relevant_texts = search_faiss_index(faiss_index, metadata, prompt, k=3, threshold=20.0)
        if relevant_texts:
            context_from_faiss = "\n\nRelevant information found in knowledge base:\n---\n" + "\n---\n".join(relevant_texts) + "\n---"
            logger.info("Added context from FAISS search to the prompt.")

    # 2. Construct the full prompt
    system_prompt = get_system_prompt() # Get base system prompt with data.txt content
    full_prompt = system_prompt + context_from_faiss + "\n\nUser: " + prompt + "\n\nAssistant:"
    logger.debug(f"Full prompt length: {len(full_prompt)}")

    # 3. Call the Gemini API
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        logger.info(f"Sending request to Gemini model: {GEMINI_MODEL_NAME}")
        response = model.generate_content(contents=[full_prompt]) # Send as list for potential future multi-turn

        # 4. Process the response
        if response.parts:
            assistant_response = response.text
            logger.info("Received successful response from Gemini.")
        # Check for blocked content or empty candidates
        elif not response.candidates or response.candidates[0].finish_reason.name != "STOP":
            block_reason = "Unknown"
            safety_ratings_str = "N/A"
            try:
                # Try to access feedback and safety ratings safely
                 if response.prompt_feedback and response.prompt_feedback.block_reason:
                     block_reason = response.prompt_feedback.block_reason.name
                 if response.candidates and response.candidates[0].safety_ratings:
                     safety_ratings_str = ", ".join([f"{r.category.name}: {r.probability.name}" for r in response.candidates[0].safety_ratings])

                 assistant_response = f"⚠️ The response was blocked. Reason: {block_reason}. Safety Ratings: [{safety_ratings_str}]"
                 logger.warning(f"Gemini response blocked. Reason: {block_reason}, Safety: [{safety_ratings_str}]")
                 st.warning(f"Response may have been blocked due to safety settings ({block_reason}).")

            except Exception as feedback_err:
                 assistant_response = "⚠️ The response was blocked, but details could not be retrieved."
                 logger.warning(f"Gemini response blocked, error getting details: {feedback_err}")
                 st.warning("Response may have been blocked due to safety settings.")

        else: # Unexpected structure
            assistant_response = "Received an unexpected response structure from the AI."
            logger.warning(f"Unexpected Gemini response structure: {response}")
            st.warning(f"Unexpected response structure: {response}")

    except Exception as e:
        logger.error(f"❌ Error generating content with Gemini: {e}")
        st.error(f"❌ Error generating AI response: {e}")
        assistant_response = f"Sorry, an error occurred while contacting the AI: {e}"

    return assistant_response