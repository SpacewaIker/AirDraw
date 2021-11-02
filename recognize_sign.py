import mediapipe as mp
import pickle
import cv2
import pyvirtualcam
import sklearn.ensemble as ensemble

with open('hand_recognition_model.pkl', 'rb') as file:
    pickle_model = pickle.load(file)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

capture = cv2.VideoCapture(0)
fmt = pyvirtualcam.PixelFormat.BGR
FPS = 20

FONT = cv2.FONT_HERSHEY_SIMPLEX

with pyvirtualcam.Camera(width=1280, height=720, fps=FPS, fmt=fmt) as camera:
    with mp_hands.Hands(static_image_mode=False,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7) as hands:
        while True:
            ret_val, image = capture.read()

            image = cv2.resize(image, (1280, 720))
            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_height, image_width, _ = image.shape
            annotated_image = image.copy()

            results = hands.process(image)
            mh_landmarks = results.multi_hand_landmarks

            if mh_landmarks:
                for hand_landmarks in mh_landmarks:
                    # Print index finger tip coordinates.
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                handedness = results.multi_handedness[0].classification[0].label

                text_x = int(mh_landmarks[0].landmark[0].x * capture.get(3))
                text_y = int(mh_landmarks[0].landmark[0].y * capture.get(4))

                # print(handedness, end='')
                if handedness == 'Right':
                    list_tuples = [(1-i.x, i.y, i.z) for i in mh_landmarks[0].landmark]
                if handedness == 'Left':
                    list_tuples = [(i.x, i.y, i.z) for i in mh_landmarks[0].landmark]

                features = [[i for t in list_tuples for i in t]]

                pred = pickle_model.predict(features)
                # print(pred)

                annotated_image = cv2.putText(
                    annotated_image,
                    str(pred[0]),
                    (text_x, text_y),
                    FONT, 3, (0, 0, 0), 5, cv2.LINE_AA)

            cv2.imshow('Preview',
                       cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            camera.send(annotated_image)
            camera.sleep_until_next_frame()

            cv2.waitKeyEx(int(1000/FPS))
