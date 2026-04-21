import os
from face_recognition_knn_classifier import knnModel


# train_dir = os.path.join("../", 'persons')
# model_path = os.path.join("../", 'model', 'test_save.clf')
base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(base_dir, '..', 'persons')
model_path = os.path.join(base_dir, '..', 'model', 'trained_knn_model.clf')
os.makedirs(os.path.dirname(model_path), exist_ok=True)

print("Starting training...")

knn_clf = knnModel(
    train_dir=train_dir,
    model_save_path=model_path,
    n_neighbors=None,
    detection_model="hog",
    number_of_times_to_upsample=0,
    use_image_enhancement=True,
)

print("Training complete!")



