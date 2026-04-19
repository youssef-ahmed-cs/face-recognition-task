"""
Sample client for Face Recognition API.
Demonstrates how to use the API endpoints.
"""

import requests
import json
from pathlib import Path


class FaceRecognitionClient:
    """Client for Face Recognition API."""
    
    def __init__(self, base_url="https://face-recognition-v01.up.railway.app/"):
        self.base_url = base_url
    
    def health_check(self):
        """Check if the API is running."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def predict(self, image_path, distance_threshold=0.5):
        """
        Predict face identities from an image.
        
        Args:
            image_path: Path to image file
            distance_threshold: Confidence threshold (0-1)
        
        Returns:
            Dictionary with detections
        """
        with open(image_path, "rb") as f:
            files = {"file": f}
            params = {"distance_threshold": distance_threshold}
            response = requests.post(
                f"{self.base_url}/predict",
                files=files,
                params=params
            )
        
        response.raise_for_status()
        return response.json()
    
    def train(self):
        """Train the model using current training data."""
        response = requests.post(f"{self.base_url}/train")
        response.raise_for_status()
        return response.json()
    
    def get_info(self):
        """Get model and training data information."""
        response = requests.get(f"{self.base_url}/info")
        response.raise_for_status()
        return response.json()


def main():
    """Example usage of the API client."""
    
    # Initialize client
    client = FaceRecognitionClient("https://face-recognition-v01.up.railway.app/")
    
    print("=" * 60)
    print("Face Recognition API Client")
    print("=" * 60)
    
    # 1. Check health
    print("\n1. Checking API health...")
    try:
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Model exists: {health['model_exists']}")
        print(f"   Training data exists: {health['training_data_exists']}")
    except requests.exceptions.ConnectionError:
        print("   ERROR: Could not connect to API. Make sure it's running!")
        return
    
    # 2. Get model info
    print("\n2. Getting model information...")
    info = client.get_info()
    print(f"   Model exists: {info['model_exists']}")
    print(f"   Training classes: {info['training_classes']}")
    
    # 3. Train model (if needed)
    if not info['model_exists']:
        print("\n3. Training model (not found)...")
        try:
            result = client.train()
            print(f"   {result['message']}")
        except Exception as e:
            print(f"   ERROR: {e}")
            return
    else:
        print("\n3. Model already trained. Skipping training.")
    
    # 4. Make predictions
    print("\n4. Making predictions...")
    
    # Try to find a test image
    test_images = list(Path(".").rglob("*.jpg")) + list(Path(".").rglob("*.jpeg"))
    
    if test_images:
        test_image = str(test_images[0])
        print(f"   Using test image: {test_image}")
        
        try:
            results = client.predict(test_image, distance_threshold=0.5)
            
            print(f"   Total faces detected: {results['total_faces']}")
            
            for i, detection in enumerate(results['detections'], 1):
                print(f"\n   Detection {i}:")
                print(f"     Name: {detection['name']}")
                print(f"     Confidence: {detection['confidence']:.2%}")
                bbox = detection['bounding_box']
                print(f"     Bounding box: ({bbox['left']}, {bbox['top']}) to ({bbox['right']}, {bbox['bottom']})")
        
        except Exception as e:
            print(f"   ERROR: {e}")
    else:
        print("   No test images found. Provide an image path for testing.")
        print("   Example: python -c \"from client import FaceRecognitionClient; client = FaceRecognitionClient(); print(client.predict('path/to/image.jpg'))\"")


if __name__ == "__main__":
    main()
