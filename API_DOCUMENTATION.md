# Face Recognition FastAPI

A FastAPI-based web service for face recognition using KNN classifier.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Training Data

Place your training images in the `persons/` directory with subdirectories for each person:

```
persons/
  ├── person1/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  ├── person2/
  │   ├── image1.jpg
  │   └── ...
```

Each subdirectory should contain at least a few images of the same person.

### 3. Train the Model

Run the training script to generate the model:

```bash
python src/train.py
```

This creates `model/trained_knn_model.clf`.

### 4. Start the API Server

```bash
python src/app.py
```

Or with uvicorn:

```bash
uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Health Check

**GET** `/health`

Check if the API is running and if the model is available.

**Response:**
```json
{
  "status": "healthy",
  "model_exists": true,
  "training_data_exists": true
}
```

---

### 2. Predict

**POST** `/predict`

Identify faces in an uploaded image.

**Parameters:**
- `file` (multipart/form-data): Image file (JPG, PNG, etc.)
- `distance_threshold` (query, optional): Confidence threshold 0-1, default=0.5

**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "name": "person1",
      "confidence": 0.92,
      "bounding_box": {
        "top": 50,
        "right": 200,
        "bottom": 250,
        "left": 100
      }
    },
    {
      "name": "unknown",
      "confidence": 0.45,
      "bounding_box": {
        "top": 100,
        "right": 350,
        "bottom": 300,
        "left": 200
      }
    }
  ],
  "total_faces": 2
}
```

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/image.jpg" \
  -F "distance_threshold=0.5"
```

**Example using Python:**
```python
import requests

with open("image.jpg", "rb") as f:
    files = {"file": f}
    params = {"distance_threshold": 0.5}
    response = requests.post("http://localhost:8000/predict", files=files, params=params)
    print(response.json())
```

---

### 3. Train Model

**POST** `/train`

Retrain the model using the current training data in the `persons/` directory.

**Response:**
```json
{
  "success": true,
  "message": "Model trained successfully",
  "model_path": "/path/to/model/trained_knn_model.clf"
}
```

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/train"
```

---

### 4. Get Info

**GET** `/info`

Get information about the model and training data.

**Response:**
```json
{
  "model_path": "/path/to/model/trained_knn_model.clf",
  "model_exists": true,
  "training_dir": "/path/to/persons",
  "training_dir_exists": true,
  "training_classes": ["person1", "person2", "person3"]
}
```

---

## Interactive API Documentation

FastAPI provides interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

You can test all endpoints directly from these interfaces.

## Configuration

Edit [src/app.py](src/app.py) to modify:

- `MODEL_PATH`: Path to the trained model
- `TRAIN_DIR`: Path to training data directory
- API host and port (in `__main__` section)

## Performance Optimization

The FastAPI server uses these optimizations:

- **Model Caching**: The trained model is cached in memory for faster predictions
- **HOG Detection**: Uses HOG (Histogram of Oriented Gradients) instead of CNN for faster face detection
- **No Image Enhancement**: Disabled by default for speed; enable if needed
- **CORS Enabled**: Allows cross-origin requests for frontend integration

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Successful prediction/operation
- `400`: Invalid image format or processing error
- `404`: Model or training data not found
- `500`: Server error during training

## Example Workflow

```bash
# 1. Prepare training data in persons/ directory
# 2. Train the model
python src/train.py

# 3. Start the API
python src/app.py

# 4. Make predictions via API
curl -X POST "http://localhost:8000/predict" -F "file=@test.jpg"

# 5. Visit http://localhost:8000/docs for interactive API testing
```

## Notes

- Ensure each person in training data has at least 5-10 images
- Images should be clear and show faces clearly
- Larger training datasets improve accuracy
- The model uses KNN (k-Nearest Neighbors) classifier
- Confidence threshold controls sensitivity: higher = stricter matching
