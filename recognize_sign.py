import mediapipe as mp
import pickle
import cv2
import pyvirtualcam
import sklearn.ensemble as ensemble
import numpy as np
import pandas as pd
import os

with open('hand_recognition_model.pkl', 'rb') as file:
    pickle_model = pickle.load(file)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

capture = cv2.VideoCapture(0)
fmt = pyvirtualcam.PixelFormat.BGR
FPS = 20

FONT = cv2.FONT_HERSHEY_SIMPLEX

DIR = 'Datapoint_files/'   # directory for the datapoint files
DEBUG = False

last = ''
CONTROLS = ['', 'Move', 'Delete']
SHAPES = ['Ellipse', 'Rectangle', 'Triangle', 'Line']
collecting = False

# Check for previous datapoint files
i = 1
while True:
    if os.path.isfile(DIR + f'dp_{i}.csv'):
        i += 1
        continue

    file_num = i
    break


def predict_(feats, last, min_confidence):
    global pickle_model
    probs = pickle_model.predict_proba(feats)[0]

    largest_prob = np.max(probs)
    index = np.where(probs == largest_prob)[0][0]
    prediction = ['Delete', 'Ellipse', 'Line', 'Move',
                  'Rectangle', 'Triangle'][index]

    # Checks that the probability is larger than the minimal confidence
    # or that the prediction is equal to the last (this avoids "flickering")
    if (largest_prob > min_confidence) or (prediction == last):
        return prediction

    return ''


with pyvirtualcam.Camera(width=1280, height=720, fps=FPS, fmt=fmt) as camera:
    with mp_hands.Hands() as hands:
        while True:
            ret_val, image = capture.read()

            image = cv2.resize(image, (1280, 720))
            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_height, image_width, _ = image.shape
            annotated_image = image.copy()

            results = hands.process(image)
            multi_hand_world = results.multi_hand_world_landmarks
            multi_hand = results.multi_hand_landmarks

            if multi_hand:
                if DEBUG:
                    for hand_landmarks in multi_hand:
                        # Print index finger tip coordinates.
                        mp_drawing.draw_landmarks(
                            annotated_image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                handedness = results.multi_handedness[0].classification[0].label

                # print(handedness, end='')
                if handedness == 'Right':
                    list_tuples = [(-i.x, i.y, i.z) for i in multi_hand_world[0].landmark]
                if handedness == 'Left':
                    list_tuples = [(i.x, i.y, i.z) for i in multi_hand_world[0].landmark]

                features = [[i for t in list_tuples for i in t]]

                pred = predict_(features, last, min_confidence=0.65)
                if DEBUG:
                    print(pred, end='\t')
                    print(pickle_model.predict_proba(features))

                annotated_image = cv2.putText(
                    annotated_image,
                    pred,
                    (100, 100),
                    FONT, 3, (0, 0, 0), 5, cv2.LINE_AA)

                # Datapoints collection for drawing
                if (last in CONTROLS) and (pred in SHAPES):
                    if DEBUG:
                        print('Begin collecting data')

                    datapoints = pd.DataFrame()
                    collecting = True

                if (last in SHAPES) and (pred in CONTROLS):
                    file = pd.DataFrame([last]).append(datapoints)
                    file.to_csv(
                        DIR + f'dp_{file_num}.csv', index=False)
                    file_num += 1
                    collecting = False
                    if DEBUG:
                        print('Finish collecting data')
                        print(file)

                if collecting:
                    x = multi_hand[0].landmark[9].x
                    y = multi_hand[0].landmark[9].y
                    datapoints = datapoints.append([[x, y]])

                last = pred

            cv2.imshow('Preview',
                       cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            camera.send(annotated_image)
            camera.sleep_until_next_frame()

            cv2.waitKeyEx(int(1000/FPS))
