import cv2
import face_recognition

from helpers import load_model



def recognize_from_camera(model_path, distance_threshold=0.5):

    knn_clf = load_model(model_path)

    video_capture = cv2.VideoCapture(0)

    while True:

        ret, frame = video_capture.read()
        if not ret:
            break

        # resize image to increase processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame)

        if len(face_locations) > 0:

            face_encodings = face_recognition.face_encodings(
                rgb_small_frame,
                face_locations
            )

            closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)

            are_matches = [
                closest_distances[0][i][0] <= distance_threshold
                for i in range(len(face_locations))
            ]

            for pred, loc, rec in zip(knn_clf.predict(face_encodings), face_locations, are_matches):

                name = pred if rec else "Unknown"

                top, right, bottom, left = loc

                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw square around face
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)

                # write name below face
                cv2.putText(
                    frame,
                    name,
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


recognize_from_camera("model/trained_knn_model.clf")