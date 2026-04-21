import os
import tempfile
import importlib
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from helpers import enhance_image, load_model

# Initialize FastAPI app
app = FastAPI( #create FastAPI instance with metadata
    title="Face Recognition API",
    description="API for face recognition using KNN classifier",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware( # allow to accept requests from any origin (for development purposes)
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
base_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(base_dir, '..', 'model', 'trained_knn_model.clf')
TRAIN_DIR = os.path.join(base_dir, '..', 'persons')
CLIENT_HTML_PATH = os.path.join(base_dir, 'client.html')

# Global model cache
_model_cache = None # Cached model instance to avoid reloading on every request
_face_recognition_module = None


def get_face_recognition_module(): # load face_recognition module on demand to handle missing dependencies gracefully
    global _face_recognition_module

    if _face_recognition_module is None:
        try:
            _face_recognition_module = importlib.import_module("face_recognition")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"face_recognition import failed: {str(e)}",
            )

    return _face_recognition_module


def get_model():
    """Load model from cache or disk."""
    global _model_cache
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Please train the model first."
        )
    
    if _model_cache is None:
        try:
            _model_cache = load_model(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {str(e)}")
    
    return _model_cache


def clear_model_cache():
    """Clear the cached model."""
    global _model_cache
    _model_cache = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    dependency_ok = True
    dependency_error = None
    model_load_error = None

    try:
        get_face_recognition_module()
    except HTTPException as e:
        dependency_ok = False
        dependency_error = e.detail

    try:
        get_model()
    except FileNotFoundError as e:
        model_load_error = str(e)
    except Exception as e:
        model_load_error = f"Model loading failed: {str(e)}"

    return {
        "status": "healthy",
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "model_load_error": model_load_error,
        "training_data_exists": os.path.exists(TRAIN_DIR),
        "face_recognition_available": dependency_ok,
        "face_recognition_error": dependency_error,
    }


@app.get("/")
async def root():
    """Serve the built-in web client on the app root."""
    if not os.path.exists(CLIENT_HTML_PATH):
        raise HTTPException(status_code=404, detail="client.html not found")
    return FileResponse(CLIENT_HTML_PATH, media_type="text/html")


@app.get("/client")
async def client_page():
    """Alias route for the web client page."""
    return await root()


@app.get("/client.html")
async def client_html():
    """Direct route to client.html."""
    return await root()


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    distance_threshold: float = Query(0.5, ge=0.0, le=1.0),
):

    try:
        knn_clf = get_model() # load model (cached or from disk) and handle errors gracefully
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load error: {str(e)}")

    try:
        face_recognition = get_face_recognition_module()
    except HTTPException as e:
        raise e
    
    # Save uploaded file temporarily
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
        
        # Load and process image
        try:
            image = face_recognition.load_image_file(tmp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")
        
        try:
            image = enhance_image(image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image enhancement failed: {str(e)}")
        
        # Detect faces
        try:
            face_locations = face_recognition.face_locations(image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Face detection failed: {str(e)}")
        
        if len(face_locations) == 0: # No faces detected
            return {
                "success": True,
                "detections": [],
                "total_faces": 0,
                "recognized_faces": 0,
                "unique_users_count": 0,
                "unique_users": [],
                "message": "No faces detected in the image",
            }
        
        # Get face encodings
        try:
            face_encodings = face_recognition.face_encodings(image, face_locations) # embeddings for detected faces
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Face encoding failed: {str(e)}")

        if len(face_encodings) == 0:
            return {
                "success": True,
                "detections": [],
                "total_faces": 0,
                "recognized_faces": 0,
                "unique_users_count": 0,
                "unique_users": [],
                "message": "Faces detected but could not generate encodings",
            }
        
        # Predict using KNN
        try:
            closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
            predictions = knn_clf.predict(face_encodings)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"KNN prediction failed: {str(e)}")
            
        are_matches = [
            closest_distances[0][i][0] <= distance_threshold 
            for i in range(len(face_locations))
        ]
        
        # Format results
        detections = []
        for pred, loc, is_match, distance in zip(
            predictions, face_locations, are_matches, closest_distances[0]
        ):
            name = pred if is_match else "unknown"
            confidence = 1 - distance[0]  # Convert distance to confidence
            
            detections.append({
                "name": name,
                "confidence": float(confidence),
                "bounding_box": {
                    "top": int(loc[0]),
                    "right": int(loc[1]),
                    "bottom": int(loc[2]),
                    "left": int(loc[3]),
                },
            })

        recognized_faces = [d for d in detections if d["name"] != "unknown"]
        unique_users = sorted({d["name"] for d in recognized_faces})
        
        return {
            "success": True,
            "detections": detections,
            "total_faces": len(detections),
            "recognized_faces": len(recognized_faces),
            "unique_users_count": len(unique_users),
            "unique_users": unique_users,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass


@app.get("/info")
async def get_info():
    """Get information about the current model and training data."""
    info = {
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "training_dir": TRAIN_DIR,
        "training_dir_exists": os.path.exists(TRAIN_DIR),
        "training_classes": [],
    }
    
    if os.path.exists(TRAIN_DIR):
        info["training_classes"] = [
            d for d in os.listdir(TRAIN_DIR)
            if os.path.isdir(os.path.join(TRAIN_DIR, d))
        ]
    
    return info


if __name__ == "__main__":
    import uvicorn # serve the app using uvicorn ASGI server
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
