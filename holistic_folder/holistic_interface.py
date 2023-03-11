from help_functions import *

def create_train_set_csv(file_name,landmarks):
        with open(file_name, mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(landmarks)

def process_video_stream(holistic, file_name=None, class_name=None):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        image = recolor_image(frame) # Recolor Feed
        results, face_landmarks, right_hand_landmarks, left_hand_landmarks, pose_landmarks = process_image(holistic, image) # Make Detections
        image = recolor_image_back(image) # Recolor image back to BGR for rendering

        draw_face_landmarks(image, face_landmarks) # 1. Draw face landmarks
        draw_right_hand_landmarks(image, right_hand_landmarks) # 2. Right hand
        draw_left_hand_landmarks(image, left_hand_landmarks) # 3. Left Hand
        draw_pose_landmarks(image, pose_landmarks) # 4. Pose Detections

        if file_name and class_name:
            # Export coordinates
            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                # Extract Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
                # Concate rows
                row = pose_row+face_row
                # Append class name
                row.insert(0, class_name)
                # Export to CSV
                with open(file_name, mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)
            except:
                pass

        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return results

def make_some_detections():
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        return process_video_stream(holistic)

def export_coordinates_to_train_csv(file_name, class_name):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        process_video_stream(holistic, file_name, class_name)


# running the code
def run_this_code():
    results = make_some_detections()

    num_coords = len(results.pose_landmarks.landmark)+len(results.face_landmarks.landmark)
    landmarks = ['class']
    for val in range(1, num_coords+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

    create_train_set_csv('coords.csv', landmarks)
    export_coordinates_to_train_csv("coords.csv", "happy")

# run_this_code()