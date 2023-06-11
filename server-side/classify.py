import numpy as np
import cv2
import mediapipe as mp
import os
import csv
import pandas as pd
from tensorflow import keras
import time

gesture_names = ['angry', 'bored', 'disgust',
                 'happy', 'sad', 'shy', 'stressed', 'surprised']


def get_gesture_name(prediction):

    index = np.argmax(prediction)

    return gesture_names[index]


def create_csv_file_coords(file):
    if not os.path.isfile(file):
        header = ['timestamp', 'label']
        for i in range(1, 544):
            header += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']
        with open(file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)


def create_csv_file_confidence(file):
    if not os.path.isfile(file):
        header = ['timestamp', 'label']
        header += gesture_names
        with open(file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)


def extract_landmarks(landmarks):
    if landmarks:
        return list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks.landmark]).flatten())
    else:
        return [0] * 84


def process_video_to_csv(input_file, model_path, output_file_coords, output_file_confidence, show_cam = True):

    create_csv_file_coords(output_file_coords)
    create_csv_file_confidence(output_file_confidence)

    mp_holistic, mp_drawing = mp.solutions.holistic, mp.solutions.drawing_utils
    model = keras.models.load_model(model_path)

    fps = 0

    with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        cap = cv2.VideoCapture(input_file)
        fps = cap.get(cv2.CAP_PROP_FPS)

        coords_list = []
        confidence_list = []
        start_time = time.time()
        loop_counter = 0

        while True:
            success, image = cap.read()
            if not success:
                break

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape

            results = holistic.process(image)

            face_row = extract_landmarks(results.face_landmarks)
            left_hand_row = extract_landmarks(results.left_hand_landmarks)
            right_hand_row = extract_landmarks(results.right_hand_landmarks)
            pose_row = extract_landmarks(results.pose_landmarks)

            feature_vector = face_row + pose_row + left_hand_row + right_hand_row

            if results.face_landmarks:
                if input_file == 0:
                    current_time = time.time()
                    timestamp = current_time - start_time
                else:
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

                feature_array = np.array(feature_vector).reshape((1, -1))
                prediction = model.predict(feature_array)
                gesture = get_gesture_name(prediction)

                feature_vector = [timestamp, gesture] + feature_vector
                coords_list.append(feature_vector)

                confidence = np.array(prediction).flatten()
                confidence = [format(float(num), '.2f') for num in confidence]
                confidence = [timestamp, gesture] + list(confidence)
                confidence_list.append(confidence)

                if show_cam:
                    image = cv2.putText(image, gesture + ' {:.2f}'.format(max(prediction.flatten())), (10, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    cv2.imshow('Holistic Model', image)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            loop_counter += 1

        cap.release()
        cv2.destroyAllWindows()

        df_coords = pd.DataFrame(coords_list)
        df_coords.to_csv(output_file_coords, mode='a', index=False, header=False)
        df_coords = pd.read_csv(output_file_coords)
        df_confidence = pd.DataFrame(confidence_list)
        df_confidence.to_csv(output_file_confidence, mode='a', index=False, header=False)
        df_confidence = pd.read_csv(output_file_confidence)

        return fps, df_coords, df_confidence
