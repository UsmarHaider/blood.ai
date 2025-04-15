# app.py
import streamlit as st
from PIL import Image
import numpy as np
import os
import time
import google.generativeai as genai
import dotenv

# --- Import Modules ---
import config
import model_utils
import chatbot
import faiss_utils
import web_scraper
import hospital_search
import utils

# --- Load Environment Variables ---
# Ensure this runs early, especially if AgentPro relies on env vars
dotenv.load_dotenv()

# --- Basic Setup ---
logger = config.setup_logging() # Setup logging first
logger.info("Starting Streamlit Application...")

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title=config.APP_TITLE,
    layout=config.PAGE_LAYOUT,
    initial_sidebar_state=config.INITIAL_SIDEBAR_STATE
)

# --- Initialize Google AI SDK ---
try:
    genai.configure(api_key=config.GOOGLE_API_KEY)
    logger.info("Google AI SDK configured successfully.")
except Exception as e:
    logger.error(f"Fatal error configuring Google AI SDK: {e}")
    st.error(f"❌ Critical Error: Could not configure Google AI SDK. Please check API Key and configuration. Error: {e}")
    st.stop() # Stop execution if SDK fails

# --- Load TensorFlow Models ---
# Use functions from model_utils
blood_disease_model = model_utils.load_tf_model(config.BLOOD_DISEASE_MODEL_PATH)
cell_type_model = model_utils.load_tf_model(config.CELL_TYPE_MODEL_PATH)

# --- Load/Initialize FAISS Index ---
# Use functions from faiss_utils
faiss_index, metadata = faiss_utils.load_faiss_index()
if faiss_index is None:
    faiss_index, metadata = faiss_utils.create_new_faiss_index(config.EMBEDDING_DIMENSION)

# --- Initialize Hospital Search Agent ---
# The import in hospital_search.py handles initialization and placeholders
agent = hospital_search.agent # Get the initialized agent (or None)
hospital_search_available = hospital_search.hospital_search_available # Get the status

# --- Streamlit App Title ---
st.title(f"🩸 {config.APP_TITLE}")

# ==============================================================================
#                              CHATBOT SECTION
# ==============================================================================
st.header("💬 AI Chat Assistant")
st.caption("Ask questions about blood cells and related diseases (general information only).")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    logger.debug("Initialized chat messages in session state.")

# Display past chat messages
# Use a container for better control if needed, especially for scrolling
chat_container = st.container(height=400) # Set a height for scrollability
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Get user input using st.chat_input
if prompt := st.chat_input("Your question..."):
    logger.info(f"User input received: {prompt[:50]}...")
    # Add user message to history and display it immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container: # Display in the container
        with st.chat_message("user"):
            st.markdown(prompt)

    # Generate and display assistant response
    with st.spinner("Thinking..."):
        # Call the chatbot response function, passing the FAISS index and metadata
        assistant_response = chatbot.generate_chatbot_response(prompt, faiss_index, metadata)

    # Add assistant response to history and display it
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    # Rerun to display the new assistant message in the container
    # st.rerun() # Usually needed, but sometimes causes issues with chat_input. Test carefully.
    # Manually display the last message if rerun is problematic:
    with chat_container:
        with st.chat_message("assistant"):
             st.markdown(assistant_response)
    # Consider using experimental_rerun() if needed, or structure to avoid rerun if possible.
    # For simple chat, displaying manually after generation might be sufficient.


# ==============================================================================
#                       IMAGE CLASSIFICATION SECTION
# ==============================================================================
st.divider()
st.header("🔬 Blood Image Classifier")

# Check if models loaded successfully before proceeding
if blood_disease_model is None and cell_type_model is None:
    st.error("❌ Both classification models failed to load. Image classification is unavailable.")
elif blood_disease_model is None:
     st.warning("⚠️ Blood Disease model failed to load. Only Cell Type classification is available.")
elif cell_type_model is None:
     st.warning("⚠️ Cell Type model failed to load. Only Blood Disease classification is available.")


# Only show the interface if at least one model is loaded
if blood_disease_model is not None or cell_type_model is not None:
    col1, col2 = st.columns([1, 2]) # Left column for controls, right for image/results

    with col1:
        # Determine available classification options
        classification_options = []
        if blood_disease_model is not None:
            classification_options.append("Blood Disease Indicator")
        if cell_type_model is not None:
            classification_options.append("Blood Cell Type")

        if not classification_options:
             st.error("Internal Error: No models available despite passing initial check.") # Should not happen
        else:
            option = st.selectbox(
                "Choose classification mode:",
                classification_options,
                key="classification_type",
                help="Select the type of analysis to perform on the uploaded image."
            )

            # Conditional UI elements based on selection
            location_for_search = None
            if option == "Blood Disease Indicator":
                st.info("This model predicts potential genetic markers (like NPM1, PML_RARA, RUNX1_RUNX1T1) or 'control'. This is NOT a diagnosis.")
                location_for_search = st.selectbox(
                    "Select location for potential hospital search:",
                    config.AVAILABLE_LOCATIONS,
                    key="location_selector",
                    help="Select your location to find nearby hospitals potentially relevant to the detected indicator after classification."
                )
            elif option == "Blood Cell Type":
                 st.info("This model identifies the type of blood cell shown (e.g., lymphocyte, neutrophil).")


            uploaded_file = st.file_uploader(
                "Upload a blood cell image",
                type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'], # Allow common image types
                key="file_uploader"
            )

    with col2:
        st.markdown("##### Classification Result") # Add a subheader for clarity
        result_placeholder = st.empty() # Placeholder for results/image

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                logger.info(f"Image uploaded: {uploaded_file.name}, size: {image.size}, mode: {image.mode}")
                # Display uploaded image using the placeholder
                result_placeholder.image(image, caption="Uploaded Image", use_column_width='auto') # Use auto for better scaling

                predicted_class = None
                confidence = None
                model_to_use = None
                class_names_to_use = None
                target_size_to_use = None

                # Select model based on user choice
                if option == "Blood Disease Indicator" and blood_disease_model:
                    model_to_use = blood_disease_model
                    class_names_to_use = config.DISEASE_INDICATOR_CLASS_NAMES
                    target_size_to_use = config.BLOOD_DISEASE_TARGET_SIZE
                elif option == "Blood Cell Type" and cell_type_model:
                    model_to_use = cell_type_model
                    class_names_to_use = config.CELL_TYPE_CLASS_NAMES
                    target_size_to_use = config.CELL_TYPE_TARGET_SIZE

                if model_to_use and class_names_to_use and target_size_to_use:
                    with st.spinner(f"Analyzing image for {option}..."):
                        predicted_class, confidence = model_utils.preprocess_and_predict(
                            image, model_to_use, class_names_to_use, target_size_to_use
                        )

                    # Display results below the image in the same column
                    if predicted_class is not None and confidence is not None:
                        st.success(f"Predicted {option}: **{predicted_class}**")
                        st.metric(label="Confidence", value=f"{confidence:.2f}%")

                        # Display additional info/actions based on prediction type
                        description = utils.get_disease_description(predicted_class) # Get description for any class
                        st.info(f"**About '{predicted_class}':** {description}")

                        # Hospital search section (only if Blood Disease Indicator was predicted)
                        if option == "Blood Disease Indicator" and location_for_search:
                             st.markdown("---") # Separator
                             expander_title = f"Search for Hospitals related to '{predicted_class}' in {location_for_search}"
                             with st.expander(expander_title, expanded=False): # Default to collapsed
                                 if not hospital_search_available:
                                     st.warning("Hospital search functionality is not available (AgentPro might be missing or failed to initialize).")
                                 elif predicted_class.lower() == 'control':
                                     st.info("Hospital search is typically not needed for 'control' results in this context.")
                                 else:
                                     if st.button(f"Search Hospitals in {location_for_search}", key=f"search_{predicted_class}"):
                                         with st.spinner(f"Searching for hospitals specializing in '{predicted_class}' in {location_for_search}..."):
                                             hospital_results = hospital_search.search_hospitals(predicted_class, location_for_search)
                                             st.markdown("##### Hospital Search Results")
                                             st.markdown(hospital_results, unsafe_allow_html=True) # Allow markdown formatting from agent

                    else:
                        st.warning("Could not make a prediction for the uploaded image.")
                else:
                     st.error("Selected model is not available. Cannot perform classification.")

            except Exception as e:
                logger.error(f"Error handling uploaded image: {e}")
                st.error(f"❌ Error processing uploaded image: {e}")
                result_placeholder.empty() # Clear the image if processing fails

        else:
            # Show message if no file is uploaded yet
             result_placeholder.info("👆 Upload an image using the panel on the left to start classification.")

# ==============================================================================
#                       WEB SCRAPING & KNOWLEDGE BASE SECTION
# ==============================================================================
st.divider()
st.header("🌐 Knowledge Base Management")
st.caption("Add information from web pages to the chatbot's knowledge base.")

scraper_col1, scraper_col2 = st.columns([2, 1]) # Give more space to input/results

with scraper_col1:
    st.subheader("Add Content from URL")
    user_url = st.text_input("Enter URL:", key="url_input", placeholder="e.g., https://www.cancer.gov/types/leukemia")

    if st.button("Extract & Add to Knowledge Base", key="scrape_save_button"):
        if user_url:
            with st.spinner(f"Processing URL: {user_url}..."):
                # 1. Scrape content
                logger.info(f"User initiated scraping for URL: {user_url}")
                scrape_success, scraped_content_or_error = web_scraper.scrape_website_content(user_url)

                if scrape_success:
                    st.text_area("Preview of Extracted Text:", scraped_content_or_error, height=150)
                    # 2. Save to knowledge base and create embeddings
                    # Pass the global faiss_index and metadata objects
                    save_success, save_message = web_scraper.save_content_to_knowledge_base(
                        user_url,
                        scraped_content_or_error,
                        faiss_index, # Pass the index object
                        metadata     # Pass the metadata dict
                    )
                    if save_success:
                        st.success(save_message)
                        # Reload chatbot prompt context if needed? Or assume it loads on next interaction
                        logger.info(f"Knowledge base updated successfully from {user_url}")
                    else:
                        st.error(f"Failed to fully update knowledge base: {save_message}")
                        logger.error(f"Failed to update knowledge base from {user_url}: {save_message}")
                else:
                    # Show scraping error
                    st.error(f"Failed to extract content: {scraped_content_or_error}")
                    logger.error(f"Scraping failed for {user_url}: {scraped_content_or_error}")
        else:
            st.warning("Please enter a URL first.")

with scraper_col2:
    st.subheader("Knowledge Base Status")
    if st.button("Show Current Status", key="view_kb_stats"):
        with st.spinner("Analyzing knowledge base..."):
            # Display stats about the text file
            st.markdown("**Text File (`data.txt`)**")
            if os.path.exists(config.DATA_FILE_PATH):
                try:
                    with open(config.DATA_FILE_PATH, 'r', encoding='utf-8') as file:
                        content = file.read()
                    total_chars = len(content)
                    total_lines = content.count('\n') + 1
                    urls_count = content.count('--- CONTENT FROM:')
                    st.write(f"- Size: {total_chars:,} characters, {total_lines:,} lines")
                    st.write(f"- Sources Found: {urls_count} URLs")
                except Exception as e:
                    st.error(f"Error reading data.txt: {e}")
            else:
                st.info("`data.txt` file not found or empty.")

            # Display stats about the FAISS index
            st.markdown("**Search Index (FAISS)**")
            if faiss_index is not None and metadata is not None:
                if faiss_index.ntotal > 0:
                     unique_sources = len(set(metadata.get('urls', [])))
                     st.write(f"- Indexed Items: {faiss_index.ntotal:,} text chunks")
                     st.write(f"- Unique Sources: {unique_sources:,} URLs")
                     st.write(f"- Embedding Dimension: {faiss_index.d}")
                     st.write(f"- Model Used: `{config.EMBEDDING_MODEL_NAME}`")
                else:
                     st.info("FAISS index is empty.")
            else:
                st.info("FAISS index is not loaded or initialized.")

st.caption("Note: Content extraction success depends heavily on website structure and may not always work. Respect website terms of service.")


# ==============================================================================
#                           MODEL SUMMARY SECTION (Optional)
# ==============================================================================
st.divider()
st.header("⚙️ TensorFlow Model Details")

# Get the last selected classification type to show the relevant summary
selected_option_for_summary = st.session_state.get("classification_type", None) # Get from session state

with st.expander("Show TF Model Summaries"):
    st.write("**Blood Disease Indicator Model Summary:**")
    if blood_disease_model:
        summary_lines = []
        blood_disease_model.summary(print_fn=lambda x: summary_lines.append(x))
        st.text('\n'.join(summary_lines))
    else:
        st.warning("Blood Disease Indicator model (`blood_cells_model.h5`) is not loaded.")

    st.write("**Blood Cell Type Model Summary:**")
    if cell_type_model:
        summary_lines = []
        cell_type_model.summary(print_fn=lambda x: summary_lines.append(x))
        st.text('\n'.join(summary_lines))
    else:
        st.warning("Blood Cell Type model (`image_classification_model.h5`) is not loaded.")

logger.info("Streamlit application setup complete. Waiting for user interaction.")