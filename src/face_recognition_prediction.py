import os
import face_recognition

from helpers import enhance_image, load_model

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '..', 'model', 'trained_knn_model.clf')


def predict(image_path, distance_threshold=0.5):

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train.py first to create it."
        )

    knn_clf = load_model(model_path)

    image = face_recognition.load_image_file(image_path)
    image = enhance_image(image)

    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        return []

    face_encodings = face_recognition.face_encodings(image,face_locations)

    closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
    # distance and index of closest neighbor for each encoding

    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]

    # are_matches = []
    # for i in range(len(face_locations)):
    #     distance = closest_distances[0][i][0]

    #     if distance <= distance_threshold:
    #         are_matches.append(True)
    #     else:
    #         are_matches.append(False)


    predictions = []

    for pred, loc, rec in zip(knn_clf.predict(face_encodings), face_locations, are_matches):
        name = pred if rec else "unknown"
        predictions.append((name, loc))

    return predictions


if __name__ == "__main__":
    test_image_path = os.path.join(
        base_dir,
        '..',
        'persons',
        'Youssef',
        'FB_IMG_1751293757851.jpg',
    )

    results = predict(test_image_path)

    if not results:
        print("No face detected.")

    for name, _ in results:
        print("Person:", name)