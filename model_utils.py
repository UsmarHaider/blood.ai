# model_utils.py
import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st
import os
import logging

# Get logger from config or main app
logger = logging.getLogger('bloodcell_app') # Assumes logger is configured in app.py

@st.cache_resource # Caching is crucial for performance
def load_tf_model(model_path):
    """Loads a TensorFlow/Keras model, handling potential errors."""
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at path: {model_path}")
        st.error(f"Model file not found: {os.path.basename(model_path)}. Please ensure it's in the correct directory.")
        return None
    try:
        logger.info(f"Loading TensorFlow model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model {os.path.basename(model_path)} loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading TensorFlow model from {model_path}: {e}")
        st.error(f"Error loading model {os.path.basename(model_path)}: {e}")
        return None

def preprocess_and_predict(image: Image.Image, model: tf.keras.Model, class_names: list, target_size: tuple):
    """Preprocess image and predict class using the provided model."""
    if model is None:
        st.error("Prediction model is not loaded.")
        return None, None

    try:
        # Resize and ensure RGB
        img_resized = image.resize(target_size)
        if img_resized.mode != 'RGB':
            img_resized = img_resized.convert('RGB')

        # Normalize and add batch dimension
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0) # Shape: (1, height, width, 3)

        # Predict
        predictions = model.predict(img_array) # Shape: (1, num_classes)

        # Get predicted class and confidence
        pred_index = np.argmax(predictions, axis=1)[0]
        if pred_index < len(class_names):
            predicted_class = class_names[pred_index]
            confidence = np.max(predictions) * 100
            logger.info(f"Prediction: {predicted_class} with {confidence:.2f}% confidence.")
            return predicted_class, confidence
        else:
            logger.error(f"Prediction index {pred_index} is out of bounds for class_names list (length {len(class_names)}).")
            st.error("Model prediction resulted in an invalid class index. Check model compatibility.")
            return None, None

    except Exception as e:
        logger.error(f"Error during image preprocessing or prediction: {e}")
        st.error(f"Image processing/prediction error: {e}")
        return None, None