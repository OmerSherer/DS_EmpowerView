import mediapipe as mp # Import mediapipe
import cv2 # Import opencv

import csv
import numpy as np

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions


def recolor_image(image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def recolor_image_back(image):
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def draw_face_landmarks(image, face_landmarks):
    mp_drawing.draw_landmarks(image, face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )

def draw_right_hand_landmarks(image, right_hand_landmarks):
    mp_drawing.draw_landmarks(image, right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )

def draw_left_hand_landmarks(image, left_hand_landmarks):
    mp_drawing.draw_landmarks(image, left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )

def draw_pose_landmarks(image, pose_landmarks):
    mp_drawing.draw_landmarks(image, pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

def process_image(holistic, image):
    results = holistic.process(image)
    face_landmarks = results.face_landmarks
    right_hand_landmarks = results.right_hand_landmarks
    left_hand_landmarks = results.left_hand_landmarks
    pose_landmarks = results.pose_landmarks
    return results, face_landmarks, right_hand_landmarks, left_hand_landmarks, pose_landmarks