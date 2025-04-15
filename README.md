Blood.AI
Advanced Blood Cell Analysis and Healthcare Resource Finder

📋 Overview
Blood.AI is an AI-powered platform that combines computer vision, natural language processing, and internet search capabilities to provide a comprehensive blood cell analysis solution. The system can detect genetic markers associated with blood disorders, identify various blood cell types, and locate specialized hospitals.
<p align="center">
  <img src="blood.ai/images/image1.gif" alt="Blood.AI System Architecture" width="800"/>
</p>
🔬 Key Features

Blood Disease Marker Detection: Identify genetic markers (NPM1, PML_RARA, RUNX1_RUNX1T1) associated with blood disorders
Blood Cell Type Classification: Categorize different blood cell types (lymphocytes, monocytes, neutrophils, etc.)
Hospital Search: Locate specialized healthcare facilities based on detected conditions and user location
Medical AI Assistant: Get reliable information about blood-related medical topics
Knowledge Base Management: Extract and index medical information from trusted web sources

🏆 Technology Stack

Frontend: Streamlit web interface
Machine Learning: TensorFlow/Keras for image classification models
Natural Language Processing: Google Gemini API for embeddings and AI assistant
Vector Database: FAISS for efficient knowledge retrieval
Internet Search: AgentPro with AresInternetTool for healthcare facility searches

🔧 Installation
Prerequisites

Python 3.8+
pip (Python package installer)
Git

Step 1: Clone the Repository
bashgit clone https://github.com/UsmarHaider/blood.ai.git
cd blood.ai
Step 2: Create and Activate Virtual Environment
bash# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
Step 3: Install Dependencies
bashpip install -r requirements.txt
Step 4: Install AgentPro
Follow the instructions to install the AgentPro tool for hospital search capabilities:
bashgit clone https://github.com/traversaal/AgentPro.git
cd AgentPro
pip install -e .
cd ..
Step 5: Set Up Environment Variables
Create a .env file in the project root:
# OpenAI API (for AgentPro)
OPENAI_API_KEY=your_openai_api_key
TRAVERSAAL_ARES_API_KEY=your_ares_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
MODEL_NAME=gpt-4o-mini  # or your preferred model

# Google API (for Gemini models)
GOOGLE_API_KEY=your_google_api_key
Also create a .streamlit/secrets.toml file:
tomlGOOGLE_API_KEY = "your_google_api_key"
Step 6: Download Model Files
Download the pre-trained TensorFlow model files from our releases page and place them in the project root:

blood_cells_model.h5 - Disease marker detection model
image_classification_model.h5 - Cell type classification model

Alternatively, you can train your own models using the training scripts in the model_training directory.
🚀 Running the Application
Start the Streamlit application with:
bashstreamlit run app.py
The application will open in your web browser at http://localhost:8501.
💻 Usage Guide
Blood Cell Classification

Select the classification mode: "Blood Disease" or "Blood Cell Type Classification"
If in "Blood Disease" mode, select your location for hospital search
Upload a blood cell image (supported formats: PNG, JPG, JPEG, TIFF)
View the classification results:

Disease marker or cell type
Confidence percentage
For disease markers: information about the marker and option to search for hospitals



Hospital Search
After a disease marker is detected:

Expand the "Find hospitals" section
Click "Search for Specialized Hospitals"
View results showing specialized hospitals in your selected location

AI Assistant

Enter questions about blood cells, blood disorders, or related medical topics
Get educational information from the AI assistant
Note: The assistant does not provide medical advice or diagnoses

Knowledge Base Management
To add medical content to the assistant's knowledge:

Enter a URL in the "Web Content Extractor" section
Click "Extract & Save to Knowledge Base"
Review the extracted content
Use "View Current Knowledge Base Stats" to see statistics about your knowledge base

📊 System Architecture
The system consists of three primary components:

Classification System

TensorFlow models for image analysis
Pre-trained for genetic markers and cell types
Image preprocessing and prediction pipeline


Medical AI Assistant

Google Gemini-powered conversational AI
FAISS vector database for knowledge retrieval
Web scraping for knowledge base expansion


Hospital Search System

AgentPro with AresInternetTool for real-time search
Location-based search within Pakistan
Caching system for performance optimization



🔍 API Reference
Key functions and their parameters:
pythondef search_hospitals(agent, disease: str, location: str, force_refresh: bool = False) -> str:
    """Search for hospitals specializing in treating a specific blood disease."""
    
def preprocess_and_predict(image, model, class_names, target_size=(64, 64)):
    """Preprocess image and predict class using TensorFlow model."""
    
def generate_gemini_embedding(text, dimension=None):
    """Generate embedding vector for text using Google's Gemini API."""
📁 Project Structure
blood.ai/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables
├── .streamlit/            # Streamlit configuration
├── blood_cells_model.h5   # Disease marker detection model
├── image_classification_model.h5  # Cell type classification model
├── data.txt               # Knowledge base text storage
├── faiss_index.bin        # FAISS vector index file
├── faiss_metadata.pkl     # Metadata for FAISS index
├── model_training/        # Scripts for model training
├── tests/                 # Test scripts
└── docs/                  # Documentation
    └── images/            # Images for documentation
🔧 Troubleshooting
Common Issues

API Key Configuration

Ensure all API keys are correctly set in both .env and .streamlit/secrets.toml
Verify API keys have necessary permissions


Model Loading Errors

Check model file paths are correct
Verify TensorFlow version compatibility


AgentPro Integration

Ensure AgentPro is installed correctly
Check required API keys for AresInternetTool


Image Classification Issues

Ensure images are of sufficient quality
Check image dimensions match model requirements
Verify class names match model output classes



Debug Logs
The application uses Python's logging module:
pythonlogging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
Check Streamlit's console output for error messages and diagnostic information.
🔜 Future Development

Support for additional blood disease markers
Integration with electronic health records
Mobile application development
Multi-language support
Expanded geographic coverage for hospital search
Advanced visualization tools for blood cell analysis

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository
Create your feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add some amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

Please ensure your code follows the project's style guidelines and includes appropriate tests.
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
📝 Citation
If you use Blood.AI in your research, please cite:
@software{blood_ai,
  author = {Haider, Usmar},
  title = {Blood.AI: Advanced Blood Cell Analysis and Healthcare Resource Finder},
  url = {https://github.com/UsmarHaider/blood.ai},
  version = {1.0.0},
  year = {2025},
}
📧 Contact
Usmar Haider - GitHub Profile
<<<<<<< HEAD
Project Link: https://github.com/UsmarHaider/blood.ai
=======
Project Link: https://github.com/UsmarHaider/blood.ai
>>>>>>> e4a971e1c39ee34d4dd987e2c6e63325ea2d1ef3
