import os
import sys

os.environ['TF_METAL'] = '0' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
from PIL import Image
import logging
from typing import Dict, Any
from keras.models import load_model



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TAMBAHKAN TENSORFLOW + TRY CATCH
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    import keras
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    TF_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TensorFlow not available: {e}")
    try:
        import keras
        print(f"Using standalone Keras version: {keras.__version__}")
        TF_AVAILABLE = False
    except ImportError:
        raise ImportError("Neither TensorFlow nor standalone Keras available")

class_names = ["Cancer", "Non-Cancer"]

app = FastAPI(
    title="Breast Cancer Detection API", 
    version="1.0.0",
    root_path="/models" 
)


model = None

def load_model_with_fallback():
    """Load model with multiple fallback methods for macOS"""
    global model
    model_path = "breast_cancer_model.h5"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Method 1: Try with CPU only
    try:
        logger.info("Attempting to load model with CPU only...")
        if TF_AVAILABLE:
            # Force CPU usage
            with tf.device('/CPU:0'):
                model = keras.models.load_model(model_path, compile=False)
        else:
            model = keras.models.load_model(model_path, compile=False)
        logger.info("Model loaded successfully on CPU")
        return model
    except Exception as e:
        logger.warning(f"CPU loading failed: {e}")

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model prediction"""
    try:
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        img = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        img = np.array(img, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        logger.info(f"Preprocessed image shape: {img.shape}")
        return img
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"Error preprocessing image: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model_with_fallback()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        # Don't raise, allow API to start without model for debugging
        pass

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Breast Cancer Detection API is running",
        "model_loaded": model is not None,
        "tensorflow_available": TF_AVAILABLE,
        "platform": "macOS compatible version"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "class_names": class_names,
        "tensorflow_available": TF_AVAILABLE
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Predict breast cancer from uploaded image"""
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded. Check /health endpoint for status."
        )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image"
        )
    
    try:
        # Load and preprocess image
        image = Image.open(file.file)
        img = preprocess_image(image)
        
        # Make prediction on CPU
        logger.info("Making prediction...")
        if TF_AVAILABLE:
            with tf.device('/CPU:0'):
                prediction = model.predict(img, verbose=0)
        else:
            prediction = model.predict(img, verbose=0)
        
        # Process results
        class_id = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        
        # Get all class probabilities
        probabilities = {
            class_names[i]: float(prediction[0][i]) 
            for i in range(len(class_names))
        }
        
        result = {
            "predicted_class": class_names[class_id],
            "confidence": confidence,
            "probabilities": probabilities
        }
        
        logger.info(f"Prediction result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error during prediction: {str(e)}"
        )

@app.post("/reload-model")
async def reload_model():
    """Manually reload the model"""
    try:
        load_model_with_fallback()
        return {"message": "Model reloaded successfully", "model_loaded": model is not None}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload model: {str(e)}"
        )
