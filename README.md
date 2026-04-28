# Simple Face Recognition

A Python-based face recognition system using machine learning (KNN classifier) to identify and classify faces in images. This project provides both a REST API and client interfaces for face recognition tasks.

## Features

- 🎯 **Face Recognition**: Identify and classify faces using trained KNN model
- 🔄 **Image Enhancement**: Automatic image processing for better recognition accuracy
- 📁 **Batch Training**: Train the model on folders containing images of different people
- 🌐 **REST API**: FastAPI-based REST API for easy integration
- 🎨 **Web Interface**: HTML client for interactive face recognition
- 🐳 **Docker Support**: Containerized deployment ready
- 🚀 **Cloud Ready**: Configured for Railway deployment
- 📊 **Live Recognition**: Real-time face detection and recognition from video/camera

## Prerequisites

- Python 3.10+
- pip (Python package manager)
- For local camera access: Webcam or compatible video device
- For Docker: Docker installed on your system

## Installation

### Local Setup

1. **Clone or download this repository**
   ```bash
   cd "4- Simple Face Recognition"
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
├── src/
│   ├── app.py                              # FastAPI application (main API server)
│   ├── train.py                            # Model training script
│   ├── client.py                           # Python client for API
│   ├── client.html                         # Web interface for face recognition
│   ├── face_recognition_knn_classifier.py # KNN classifier implementation
│   ├── face_recognition_prediction.py     # Prediction logic
│   ├── helpers.py                         # Utility functions
│   ├── live_recognition.py                # Live video recognition
│   ├── data_augmentation.ipynb            # Jupyter notebook for data augmentation
│   └── Face Recognition/
│       └── augmented_data/                # Augmented training data
├── model/
│   └── trained_knn_model.clf              # Pre-trained KNN model
├── persons/                               # Training data directory
│   ├── Al-Shaarawy/
│   ├── Elon/
│   ├── Moamn/
│   └── Youssef/
├── requirements.txt                       # Python dependencies
├── Dockerfile                             # Docker configuration
├── package.json                           # Node.js dependencies (Railway)
├── railway.json                           # Railway deployment config
├── railway.toml                           # Railway service definition
└── README.md                              # This file
```

## Usage

### 1. Running the API Server

Start the FastAPI server:

```bash
cd src
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

**Interactive API Documentation**:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### 2. Using the Web Interface

1. Open `src/client.html` in your web browser
2. Upload an image containing faces
3. The system will detect and identify faces in the image
4. Results will be displayed with confidence scores

### 3. Using the Python Client

```bash
cd src
python client.py
```

### 4. Live Face Recognition

For real-time face recognition from your webcam:

```bash
cd src
python live_recognition.py
```

## Training the Model

### Adding New Faces

1. Create a folder in the `persons/` directory with the person's name
   ```
   persons/
   └── YourName/
       ├── image1.jpg
       ├── image2.jpg
       └── image3.jpg
   ```

2. Add clear, well-lit photos of the person to the folder

3. Run the training script:
   ```bash
   cd src
   python train.py
   ```

The script will:
- Detect faces in all images
- Extract face encodings
- Train the KNN model
- Save the model to `model/trained_knn_model.clf`

### Data Augmentation (Optional)

For better results with limited images, use the data augmentation notebook:

```bash
cd src
jupyter notebook data_augmentation.ipynb
```

## API Endpoints

### GET `/`
Returns the HTML client interface

### POST `/recognize`
Recognize faces in an uploaded image

**Parameters**:
- `file` (multipart/form-data): Image file to analyze
- `distance_threshold` (float, optional): Distance threshold for recognition (default: 0.6)

**Response**:
```json
{
  "faces": [
    {
      "name": "Elon",
      "distance": 0.25,
      "confidence": 95.5,
      "location": {
        "top": 50,
        "right": 150,
        "bottom": 200,
        "left": 100
      }
    }
  ],
  "total_faces": 1
}
```

### GET `/health`
Health check endpoint

## Docker Deployment

### Build and Run Locally

```bash
# Build the image
docker build -t face-recognition .

# Run the container
docker run -p 8000:8000 face-recognition
```

### Deploy to Railway

1. Push this repository to GitHub
2. Connect your GitHub repository to Railway
3. Railway will automatically:
   - Build the Docker image
   - Deploy the application
   - Expose it publicly

Configuration is already set in `railway.toml`

## Dependencies

Key Python packages:
- **FastAPI**: Modern web framework for building APIs
- **face-recognition**: Facial recognition library
- **dlib**: Deep learning toolkit
- **scikit-learn**: Machine learning for KNN classifier
- **OpenCV**: Computer vision library
- **Pillow**: Image processing
- **numpy**: Numerical computing

See `requirements.txt` for complete list with versions.

## Performance Tips

1. **Image Quality**: Use clear, well-lit images for best results
2. **Face Size**: Ensure faces are at least 100x100 pixels
3. **Training Data**: Use 5+ images per person for better accuracy
4. **Distance Threshold**: Adjust `distance_threshold` for sensitivity
   - Lower values (0.4): More strict, fewer false positives
   - Higher values (0.8): More lenient, more recognitions

## Troubleshooting

### Import Errors
```bash
pip install --upgrade -r requirements.txt
```

### Model Not Found
Ensure `model/trained_knn_model.clf` exists. Run training script if missing:
```bash
cd src
python train.py
```

### No Faces Detected
- Check image quality and lighting
- Ensure faces are clearly visible
- Try adjusting `number_of_times_to_upsample` in training

### Out of Memory
Reduce image size or batch processing. Modify `enhance_image()` in `helpers.py`

## Example Workflow

1. **Add training data**:
   ```
   persons/John/ -> photos of John
   persons/Jane/ -> photos of Jane
   ```

2. **Train the model**:
   ```bash
   python src/train.py
   ```

3. **Test with web interface**:
   - Open `src/client.html` in browser
   - Upload test image

4. **Deploy**:
   ```bash
   docker build -t face-recognition .
   docker run -p 8000:8000 face-recognition
   ```

## License

This project is provided as-is for educational and personal use.

## Contributing

Improvements and contributions are welcome! Feel free to:
- Report issues
- Suggest enhancements
- Submit pull requests

## Support

For questions or issues:
1. Check the troubleshooting section
2. Review the FastAPI documentation at https://fastapi.tiangolo.com/
3. Check face_recognition library: https://github.com/ageitgey/face_recognition

---

**Last Updated**: April 2026
