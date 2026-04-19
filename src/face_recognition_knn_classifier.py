import math
import os
import pickle
import numpy as np
from sklearn import neighbors
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from tqdm import tqdm  # Progress bar

from helpers import enhance_image



def knnModel(
    train_dir,
    model_save_path,
    n_neighbors=None,
    detection_model="hog",
    number_of_times_to_upsample=0,
    use_image_enhancement=False,
):

    encodings_features = []
    names = []

    total_images = 0
    for class_dir in os.listdir(train_dir):
        person_path = os.path.join(train_dir, class_dir)

        if not os.path.isdir(person_path):
            continue

        total_images += len(list(image_files_in_folder(person_path)))

    with tqdm(total=total_images, desc="Training Progress", unit="img") as pbar:

        for class_dir in os.listdir(train_dir):
            person_path = os.path.join(train_dir, class_dir)

            if not os.path.isdir(person_path):
                continue

            # image_files_in_folder
            # This function scans a given folder and returns a list of image file paths only, ignoring any non-image files.
            # It simplifies data loading by automatically filtering supported image formats like JPG and PNG.
            # [
            #   "persons/ahmed/img1.jpg",
            #   "persons/ahmed/img2.png"
            # ]
            for img_path in image_files_in_folder(person_path):
                # Load the image file and convert to a numpy array to face_recognition library can process.
                image = face_recognition.load_image_file(img_path)

                if use_image_enhancement:
                    image = enhance_image(image)

                face_locations = face_recognition.face_locations(
                    image,
                    number_of_times_to_upsample=number_of_times_to_upsample,
                    model=detection_model,
                )

                #[(top, right, bottom, left)]  [(50, 200, 200, 50)]
                if len(face_locations) != 1:
                    pbar.update(1)
                    continue

                encoding = face_recognition.face_encodings(image, face_locations)[0]

                encodings_features.append(encoding)
                names.append(class_dir)

                pbar.update(1)  

    encodings_features = np.array(encodings_features)
    names = np.array(names)

    if len(encodings_features) == 0:
        raise ValueError(
            "No valid training samples found. Ensure each training image contains exactly one detectable face."
        )

    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(encodings_features))))
        print("Chosen n_neighbors:", n_neighbors)

    knn_clf = neighbors.KNeighborsClassifier(
        n_neighbors=n_neighbors,
        algorithm='ball_tree', # brute 
        weights='distance'  # uniform
    )

    knn_clf.fit(encodings_features, names)

    with open(model_save_path, 'wb') as f: # wb: write binary
        pickle.dump(knn_clf, f)

    return knn_clf