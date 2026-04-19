import pickle
from PIL import Image, ImageEnhance
import numpy as np

def enhance_image(image):
    image = Image.fromarray(image)
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    return np.array(image)


def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)
