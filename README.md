# Project Report: AI Blood Cell Analysis & Information Tool

**Date:** April 15, 2025
**Version:** 1.0
**Developed For:** Informational and Educational Purposes

---

## 1. Introduction

This report details the development and features of the "AI Blood Cell Analysis & Information Tool," a web application built using Streamlit. The primary goal of this project is to provide an integrated platform combining AI-powered chatbot capabilities for blood cell and disease information with machine learning-based image classification for blood cell images.

The application leverages Google's Gemini models for conversational AI and text embeddings, TensorFlow/Keras for image analysis, and FAISS for efficient similarity search to create a Retrieval-Augmented Generation (RAG) system for the chatbot. It also includes functionality for users to expand the chatbot's knowledge base by scraping web content and an optional feature to search for specialized medical facilities (using AgentPro, if available). The project code is available at: [https://github.com/UsmarHaider/blood.ai](https://github.com/UsmarHaider/blood.ai)

**Crucially, this tool is designed for informational and educational purposes ONLY. It does not provide medical diagnoses, interpretations, or advice. Users must consult qualified healthcare professionals for any health concerns.**

## 2. Objectives

The key objectives of this project were:

* **Develop an Interactive Chatbot:** Create an AI assistant capable of answering general questions about blood cells, common blood disorders, and related terminology, using Google Gemini.
* **Implement Image Classification:** Integrate pre-trained TensorFlow/Keras models to classify uploaded blood cell images into:
    * Specific cell types (e.g., lymphocyte, monocyte).
    * Potential disease indicators/genetic markers (e.g., NPM1, PML_RARA) or a control state.
* **Enable Knowledge Base Expansion:** Allow users to add relevant information from web pages into a persistent knowledge base (`data.txt`) and a searchable FAISS vector index.
* **Integrate Retrieval-Augmented Generation (RAG):** Enhance the chatbot's responses by retrieving relevant information from the user-populated knowledge base using FAISS and Gemini embeddings.
* **Provide Conditional Hospital Search:** Offer functionality (dependent on AgentPro availability) to search for hospitals potentially specializing in conditions related to classified disease indicators in specified locations (e.g., Lahore, Karachi).
* **Promote Responsible AI Use:** Clearly communicate the tool's limitations and emphasize that it is not a substitute for professional medical consultation.
* **Ensure Modularity and Maintainability:** Structure the codebase into logical modules for better organization and future development.

## 3. Features

The application offers the following core features:

1.  **AI Chat Assistant:**
    * Powered by Google Gemini (`gemini-1.5-flash`).
    * Answers general knowledge questions based on its training and a predefined system prompt.
    * Uses RAG: Searches a FAISS index (built from `data.txt` and scraped content) for relevant context using Gemini embeddings (`text-embedding-004`) to enhance responses.
    * Maintains chat history within the user's session.
    * Strictly adheres to safety guidelines, refusing to provide diagnoses or medical advice.

2.  **Blood Image Classifier:**
    * Allows users to upload blood cell images (`.png`, `.jpg`, `.jpeg`, etc.).
    * Provides two classification modes:
        * **Blood Disease Indicator:** Predicts potential genetic markers (or control) using `blood_cells_model.h5`.
        * **Blood Cell Type:** Identifies the cell type (e.g., lymphocyte) using `image_classification_model.h5`.
    * Displays the uploaded image, the predicted class, and the model's confidence score.
    * Provides a brief description of the predicted class/indicator.
    * Includes image preprocessing (resizing, normalization) tailored to each model.

3.  **Knowledge Base Management:**
    * Users can input a URL containing relevant information.
    * The application attempts to scrape the main textual content from the URL using `requests` and `BeautifulSoup`.
    * Successfully scraped content is appended to `data.txt`.
    * The scraped text is chunked, and embeddings are generated for each chunk using Gemini (`text-embedding-004`).
    * These embeddings and corresponding text chunks are added to a FAISS vector index (`faiss_index.bin`) and associated metadata (`faiss_metadata.pkl`).
    * Provides a button to view basic statistics about the text file and the FAISS index.

4.  **Hospital Search (Conditional):**
    * Activated only if the "Blood Disease Indicator" classification predicts a specific marker (not 'control').
    * Requires the `AgentPro` library and `AresInternetTool` to be available and correctly initialized. Gracefully handles unavailability.
    * Prompts the user to select a location (defaults include Lahore, Karachi, Islamabad, etc.).
    * Uses the AgentPro agent to perform a web search for hospitals or centers specializing in the relevant condition in the chosen location.
    * Displays the formatted search results provided by the agent.
    * Includes a strong disclaimer about verifying information directly with facilities.
    * Uses in-memory caching to avoid redundant searches.

## 4. Technical Architecture

* **Framework:** Streamlit (`streamlit`)
* **Primary Language:** Python 3.x
* **Core AI / ML Components:**
    * **LLM:** Google Gemini (`google-generativeai`) for chat and embeddings.
    * **Image Classification:** TensorFlow (`tensorflow`), Keras (via TF).
    * **Vector Search:** FAISS (`faiss-cpu` or `faiss-gpu`) for similarity search in the knowledge base.
* **Web Scraping:** `requests`, `BeautifulSoup4`
* **Agent Framework (Optional):** `AgentPro` (if installed/available).
* **Data Handling:** `numpy`, `Pillow` (PIL Fork).

**Visual Architecture Diagrams:**

* **Overview Architecture:** [View Diagram (image1.gif)](https://github.com/UsmarHaider/blood.ai/blob/main/images/image1.gif)
* **Detailed Architecture:** [View Diagram (image2.gif)](https://github.com/UsmarHaider/blood.ai/blob/main/images/image2.gif)

*(Note: The diagrams above provide visual representations of the system components and their interactions.)*

* **Configuration:** Centralized `config.py`, Streamlit Secrets (`.streamlit/secrets.toml`) for API keys, `python-dotenv` for potential local environment variables.
* **Persistence:**
    * Text Knowledge Base: `data.txt`
    * FAISS Index: `faiss_index.bin`
    * FAISS Metadata: `faiss_metadata.pkl`
* **Code Structure:** Modular design with separate files for:
    * `app.py`: Main application logic, UI layout.
    * `config.py`: Configuration variables, paths, API keys.
    * `model_utils.py`: TensorFlow model loading and prediction functions.
    * `chatbot.py`: Chatbot prompt generation, API interaction logic.
    * `faiss_utils.py`: FAISS index management, embedding generation, search functions.
    * `web_scraper.py`: Web scraping and knowledge base update functions.
    * `hospital_search.py`: AgentPro integration and hospital search logic.
    * `utils.py`: General utility functions (e.g., disease descriptions).
* **Dependency Management:** `requirements.txt`

## 5. Implementation Details

* **Model Loading:** TensorFlow models are loaded using `tf.keras.models.load_model` and cached using Streamlit's `@st.cache_resource` for efficiency.
* **Image Preprocessing:** Images are resized to the target dimensions required by each model (e.g., 224x224 or 64x64), converted to RGB, and pixel values are normalized to [0, 1].
* **Embeddings:** Gemini's `text-embedding-004` model is used via `genai.embed_content`. Different `task_type` parameters (`RETRIEVAL_QUERY` vs. `RETRIEVAL_DOCUMENT`) are used for search queries and indexing content, respectively.
* **FAISS Index:** A simple `IndexFlatL2` (Euclidean distance) index is used. Text chunks are stored in a separate metadata file (`pickle`) linked by index position.
* **RAG Implementation:** When a user chats, their prompt is embedded, FAISS is searched for relevant text chunks below a distance threshold, and these chunks are added as context to the main system prompt before calling the Gemini chat model.
* **Error Handling:** `try-except` blocks are used for critical operations like API calls, file I/O, model loading, web scraping, and AgentPro initialization. User-friendly error messages are displayed via `st.error` or `st.warning`.
* **Modularity:** Functions are grouped into separate Python files based on their purpose, improving code readability and maintainability. Configuration is centralized.
* **Session State:** Streamlit's `st.session_state` is used to maintain the chat history across user interactions.

## 6. Usage

### Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/UsmarHaider/blood.ai.git](https://github.com/UsmarHaider/blood.ai.git)
    cd blood.ai
    ```

2.  **Create Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # Linux/macOS:
    source venv/bin/activate
    # Windows:
    # venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: This installs `faiss-cpu` by default. If you have a compatible GPU and CUDA installed, you might consider installing `faiss-gpu` instead for better performance.)*

4.  **Set Up API Key:**
    * Create a directory named `.streamlit` inside the `blood.ai` folder (if it doesn't exist).
    * Inside `.streamlit`, create a file named `secrets.toml`.
    * Add your Google AI API key to this file:
        ```toml
        # .streamlit/secrets.toml
        GOOGLE_API_KEY = "YOUR_ACTUAL_GOOGLE_API_KEY_HERE"
        ```

5.  **Place Model Files:**
    * Ensure the pre-trained Keras model files are present in the main `blood.ai` directory:
        * `blood_cells_model.h5` (for disease indicator classification)
        * `image_classification_model.h5` (for cell type classification)
    * *(If your models are hosted elsewhere, download them and place them here, or modify the paths in `config.py`)*

6.  **AgentPro Setup (Optional):**
    * If you intend to use the Hospital Search feature, ensure the `AgentPro` library is correctly installed or placed within the project structure (e.g., an `AgentPro/` directory) as expected by the import logic in `hospital_search.py`.
    * AgentPro might require additional setup or environment variables. Refer to its specific documentation. If AgentPro is unavailable, the hospital search feature will be disabled gracefully.

### Running the Application

1.  **Navigate to Directory:** Make sure you are in the main `blood.ai` directory in your terminal, with the virtual environment activated.
2.  **Run Streamlit:**
    ```bash
    streamlit run app.py
    ```
3.  Your default web browser should open automatically, displaying the application interface.

### Interacting with the Application

* **Chat:** Type your questions about blood cells or related diseases into the chat input box located at the bottom of the "AI Chat Assistant" section. The chat history will appear above it.
* **Classifier:**
    * Go to the "Blood Image Classifier" section.
    * Select the desired "classification mode" (Blood Disease Indicator or Blood Cell Type).
    * If classifying disease indicators, select a location for the potential hospital search.
    * Click "Browse files" or drag and drop a blood cell image onto the file uploader.
    * The results (prediction, confidence, description) will appear on the right side below the uploaded image.
    * If a disease indicator was predicted (and AgentPro is available), an expandable section will appear allowing you to click "Search Hospitals".
* **Knowledge Base:**
    * Go to the "Knowledge Base Management" section.
    * Enter a valid URL into the text input box.
    * Click "Extract & Add to Knowledge Base". The application will attempt to scrape the text, show a preview, and add it to the chatbot's searchable knowledge. Status messages will indicate success or failure.
    * Click "Show Current Status" to see statistics about the `data.txt` file and the FAISS index.

## 7. Future Enhancements

* **Advanced Scraping:** Implement more robust scraping techniques (e.g., using Selenium) to handle JavaScript-rendered content or complex site structures.
* **Improved RAG:** Explore more sophisticated chunking strategies, embedding models, re-ranking algorithms, and potentially different FAISS index types (e.g., IVF_FLAT) for larger knowledge bases.
* **Model Management:** Allow users to upload or select different classification models. Add model evaluation metrics display.
* **User Authentication:** Implement user accounts to allow for personalized chat history and knowledge bases.
* **Batch Processing:** Add functionality to classify multiple images at once.
* **UI/UX Improvements:** Refine the Streamlit interface using columns, tabs, expanders, and potentially custom components for a smoother experience.
* **Alternative Vector DBs:** Explore integration with cloud-based or other vector database solutions for scalability.
* **Direct API Integrations:** Connect to reputable medical information APIs (e.g., PubMed, UMLS) for more structured knowledge retrieval as an alternative/complement to web scraping.
* **Accessibility:** Review and improve application accessibility (WCAG compliance).

## 8. Conclusion

The AI Blood Cell Analysis & Information Tool successfully integrates multiple AI technologies into a single Streamlit application. It provides a valuable resource for learning about blood cells and related diseases through its chatbot and image classification features. The RAG implementation allows the chatbot's knowledge to be dynamically expanded by the user.

While offering powerful features, the application's limitations, particularly its inability to provide medical advice or diagnosis, are consistently emphasized. The optional hospital search feature, leveraging AgentPro, adds practical utility but requires careful user discretion and verification. The modular codebase provides a solid foundation for future development and enhancement. This tool serves as a practical demonstration of combining LLMs, computer vision, and vector search within an accessible web framework.
