from logging import PlaceHolder
import cv2
import streamlit as st
import mediapipe as mp
import pickle
import pyvirtualcam
import sklearn.ensemble as ensemble
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axisartist.axislines import Subplot
from matplotlib.patches import Circle, Rectangle, Arrow
from matplotlib.lines import Line2D
from random import uniform

st.title("Air Drawing Controls Demo")
run = st.checkbox('Run')
Outlines = st.checkbox('Hand Outlines')
capture = cv2.VideoCapture(0)


def add_shape(shape):
    global fig, last_shape
    last_shape = shape
    # shape.add(fig)
    fig.add_artist(last_shape)


def delete_shape():
    global fig, last_shape
    last_shape.remove()


col1, col2 = st.columns(2)

with col1:
    sidebartext = st.write("Video feed:")
    FRAME_WINDOW = st.image([])

# fig = plt.figure(figsize=(5, 5))
# ax = Subplot(fig, 111)
# fig.add_subplot(ax)

# ax.axis['left'].set_visible(False)
# ax.axis['bottom'].set_visible(False)
# ax.axis['right'].set_visible(False)
# ax.axis['top'].set_visible(False)

# add_shape(Circle((2.5, 2.5), 1))

with col2:
    # plot = st.pyplot(fig)
    sidebartext = st.write("Last Shape:")
    placeHolder = st.empty()

st.image("Demo/legend2.jpg", use_column_width=True)

with open('hand_recognition_model.pkl', 'rb') as file:
    pickle_model = pickle.load(file)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

fmt = pyvirtualcam.PixelFormat.BGR
FPS = 20

FONT = cv2.FONT_HERSHEY_SIMPLEX

DEBUG = False

previouspred = None


def predict_(feats, min_confidence):
    global pickle_model
    probs = pickle_model.predict_proba(feats)[0]

    largest_prob = np.max(probs)
    index = np.where(probs == largest_prob)[0][0]
    prediction = ['Delete', 'Ellipse', 'Line', 'Move',
                  'Rectangle', 'Triangle'][index]
    if largest_prob < min_confidence:
        return ''
    return prediction


with pyvirtualcam.Camera(width=1280, height=720, fps=FPS, fmt=fmt) as camera:
    with mp_hands.Hands() as hands:
        while run:
            DEBUG = Outlines

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

                # set text coordinates
                if DEBUG:
                    text_x = int(multi_hand[0].landmark[0].x * capture.get(3))
                    text_y = int(multi_hand[0].landmark[0].y * capture.get(4))
                else:
                    text_x = 500
                    text_y = 500

                # print(handedness, end='')
                if handedness == 'Right':
                    list_tuples = [(-i.x, i.y, i.z) for i in multi_hand_world[0].landmark]
                if handedness == 'Left':
                    list_tuples = [(i.x, i.y, i.z) for i in multi_hand_world[0].landmark]

                features = [[i for t in list_tuples for i in t]]

                pred = predict_(features, min_confidence=0.65)
                # if DEBUG:
                #     print(pred, end='\t')
                #     print(pickle_model.predict_proba(features))

                annotated_image = cv2.putText(
                    annotated_image,
                    pred,
                    (text_x, text_y),
                    FONT, 3, (0, 0, 0), 5, cv2.LINE_AA)
                
                if previouspred != pred:
                    if pred == 'Line':    
                        placeHolder.image('Demo/line.png', use_column_width=True)
                        # x1, y1 = uniform(0, 4), uniform(0, 4)
                        # x2, y2 = uniform(0, 4), uniform(0, 4)
                        # add_shape(Line2D([x1, x2], [y1, y2]))
                    elif pred == 'Ellipse':    
                        placeHolder.image('Demo/circle.png', use_column_width=True)
                    elif pred == 'Rectangle':
                        placeHolder.image('Demo/rectangle.png', use_column_width=True)
                    elif pred == 'Triangle':
                        placeHolder.image('Demo/triangle.png', use_column_width=True)
                    elif pred == 'Delete':
                        # delete_shape()
                        placeHolder.empty()

                    # with col2:
                    #     # plot.empty()
                    #     plot.pyplot(fig)

                previouspred = pred

            # Unmark this code for a window preview of the video feed
            #ret, frame = capture.read()
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #cv2.imshow('Preview',
                #cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            #camera.send(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            #camera.sleep_until_next_frame()

            cv2.waitKeyEx(int(1000/FPS))
            #ret, frame = capture.read()
            frame = annotated_image
            FRAME_WINDOW.image(frame)

        else:
            st.write('Stopped')
