from help_functions import *

def create_train_set_csv(file_name,landmarks):
        with open(file_name, mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(landmarks)

def process_camera(holistic, output_file=0, class_name=None, input_file=0):

    if not output_file: # in case there is no specific given output file
                    output_file = input_file.replace('.mp4', '.csv')

    if input_file:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        input_file = os.path.join(current_dir, 'videos/'+input_file)
    cap = cv2.VideoCapture(input_file)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = recolor_image(frame) # Recolor Feed
        results, face_landmarks, right_hand_landmarks, left_hand_landmarks, pose_landmarks = process_image(holistic, image) # Make Detections
        image = recolor_image_back(image) # Recolor image back to BGR for rendering

        draw_face_landmarks(image, face_landmarks) # 1. Draw face landmarks
        draw_right_hand_landmarks(image, right_hand_landmarks) # 2. Right hand
        draw_left_hand_landmarks(image, left_hand_landmarks) # 3. Left Hand
        draw_pose_landmarks(image, pose_landmarks) # 4. Pose Detections

        if class_name or 1:
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
                if class_name:
                     row.insert(0, class_name)

                # Export to CSV
                current_dir = os.path.dirname(os.path.abspath(__file__))
                output_dir = os.path.join(current_dir, 'output_csv_directory')
                if not os.path.exists(output_dir): # if the folder does not exist create it
                    os.mkdir(output_dir)

                output_file_path = output_dir + '/' + output_file
                if not os.path.exists(output_file_path): # if the file does not exist create it
                    num_coords = len(results.pose_landmarks.landmark)+len(results.face_landmarks.landmark)
                    landmarks = []
                    if class_name:
                         landmarks += ['class']
                    for val in range(1, num_coords+1):
                        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
                    create_train_set_csv(output_file_path,landmarks)
                
                with open(output_file_path, mode='a', newline='') as f: # append the coordinates to the csv
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)
            except:
                pass

        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'): # press 'q' to exist
            break

    # cap.release()
    cv2.destroyAllWindows()
    return results

def process_video_to_coordinates_csv(output_file=0, class_name=None, input_file=0):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        process_camera(holistic, output_file, class_name, input_file)

def convert_mp4_to_images(video_name):
    video_path = 'videos/'+video_name

    # Get the current working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Open the video file
    video = cv2.VideoCapture(os.path.join(current_dir, video_path))

    # Check that the video file exists
    if not os.path.isfile(os.path.join(current_dir, video_path)):
        print("Video file not found")
        exit()

    # Create a directory with the same name as the video file
    output_dir = os.path.join(current_dir, 'images/')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(current_dir, 'images/'+name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Check that the video file was opened successfully
    if not video.isOpened():
        print("Could not open video file")
        exit()

    # Create a counter variable for the output file names
    count = 0

    # Loop through the frames of the video
    while True:
        # Read the next frame
        ret, frame = video.read()

        # If there are no more frames, break out of the loop
        if not ret:
            break

        # Convert the frame to a JPG image
        output_file = os.path.join(output_dir, f'output_{count}.jpg')
        cv2.imwrite(output_file, frame)

        # Increment the counter
        count += 1

    # Release the video file
    video.release()


'''

#running the code:
#with selected class_name in every row in order to train the classifier
process_video_to_coordinates_csv(output_file='specific_output_file.csv', class_name="happy", input_file='video1.mp4') # with specific output file name
process_video_to_coordinates_csv(class_name="happy", input_file='video1.mp4') # the output file name is 'video1.csv'

#without selected class_name (only the coordinates)
process_video_to_coordinates_csv(output_file='specific_output_file.csv', input_file='video2.mp4') # with specific output file name
process_video_to_coordinates_csv(input_file='video2.mp4') # the output file name is 'video2.csv'

# filming from the webcam = without mp4 file
process_video_to_coordinates_csv(output_file='specific_output_file.csv', class_name="happy")
process_video_to_coordinates_csv(output_file='specific_output_file.csv')

# converting a video into images
convert_mp4_to_images('video1.mp4') # the images are located in a folder named images/video1
convert_mp4_to_images('video2.mp4') # the images are located in a folder named images/video2

'''