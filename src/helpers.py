import pickle
from PIL import Image, ImageEnhance
import numpy as np

def enhance_image(image):
    image = Image.fromarray(image) # Convert numpy array to PIL Image
    enhancer = ImageEnhance.Sharpness(image) # Create a sharpness enhancer use to enhance the image
    image = enhancer.enhance(2.0) # Enhance the image by a factor of 2.0 (you can adjust this value as needed)
    return np.array(image) # Convert back to numpy array for face_recognition processing


def load_model(model_path):
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    except pickle.UnpicklingError as e:
        raise ValueError(f"Model file corrupted or invalid: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

