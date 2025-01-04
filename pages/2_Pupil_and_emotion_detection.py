import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp
import plotly.express as px
import pandas as pd
from scipy.spatial import distance
import time


model = load_model('model_file_30epochs.h5')
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}


# Feature extraction function
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7
)

# Constants for thresholds
LEFT_VERTICAL_THRESHOLD = 23
LEFT_HORIZONTAL_THRESHOLD = 30
RIGHT_VERTICAL_THRESHOLD = 55
RIGHT_HORIZONTAL_THRESHOLD = 60


# Functions for pupil detection logic
def calculate_eye_distances(landmarks, indices, w, h):
    points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
    vertical = distance.euclidean(points[1], points[5])  # Top to bottom
    horizontal = distance.euclidean(points[0], points[3])  # Inner to outer
    return vertical, horizontal


# Functions for left and right eye detection
def is_left_eye_open(landmarks, indices, h, w):
    vertical, horizontal = calculate_eye_distances(landmarks, indices, w, h)
    if vertical > LEFT_VERTICAL_THRESHOLD and horizontal > LEFT_HORIZONTAL_THRESHOLD:
        return True
    else:
        return False


def is_right_eye_open(landmarks, indices, h, w):
    vertical, horizontal = calculate_eye_distances(landmarks, indices, w, h)
    if vertical > RIGHT_VERTICAL_THRESHOLD and horizontal > RIGHT_HORIZONTAL_THRESHOLD:
        return True
    else:
        return False


def get_eye_center(landmarks, indices, w, h):
    points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
    center = np.mean(points,axis=0).astype(int)
    return tuple(center)


def get_pupil_size(landmarks, indices, w, h):
    vertical, horizontal = calculate_eye_distances(landmarks, indices, w, h)
    return (vertical + horizontal) / 2


def detect_pupil(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return image, None

    for face_landmarks in results.multi_face_landmarks:
        LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 157, 173, 144]
        RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 384, 398, 373]

        h, w, _ = image.shape

        left_eye_center = get_eye_center(face_landmarks.landmark, LEFT_EYE_INDICES, w, h)
        right_eye_center = get_eye_center(face_landmarks.landmark, RIGHT_EYE_INDICES, w, h)

        left_eye_open = is_left_eye_open(face_landmarks.landmark, LEFT_EYE_INDICES, h, w)
        right_eye_open = is_right_eye_open(face_landmarks.landmark, RIGHT_EYE_INDICES, h, w)

        if not left_eye_open and not right_eye_open:
            cv2.putText(image, "Both Eyes Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return image, None

        elif left_eye_open and not right_eye_open:
            pupil_size = get_pupil_size(face_landmarks.landmark, LEFT_EYE_INDICES, w, h)
            cv2.circle(image, left_eye_center, 3, (0, 255, 0), -1)
            cv2.putText(image, f"L: {int(pupil_size)}", (left_eye_center[0] + 10, left_eye_center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(image, "Right Eye Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return image, pupil_size

        elif right_eye_open and not left_eye_open:
            pupil_size = get_pupil_size(face_landmarks.landmark, RIGHT_EYE_INDICES, w, h)
            cv2.circle(image, right_eye_center, 3, (0, 255, 0), -1)
            cv2.putText(image, f"R: {int(pupil_size)}", (right_eye_center[0] + 10, right_eye_center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(image, "Left Eye Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return image, pupil_size

        else:
            left_pupil_size = get_pupil_size(face_landmarks.landmark, LEFT_EYE_INDICES, w, h)
            right_pupil_size = get_pupil_size(face_landmarks.landmark, RIGHT_EYE_INDICES, w, h) - 44
            avg_pupil_size = (left_pupil_size + right_pupil_size) / 2

            cv2.circle(image, left_eye_center, 3, (0, 255, 0), -1)
            cv2.circle(image, right_eye_center, 3, (0, 255, 0), -1)
            cv2.putText(image, f"L: {int(left_pupil_size)}", (left_eye_center[0] + 10, left_eye_center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(image, f"R: {int(right_pupil_size)}", (right_eye_center[0] + 10, right_eye_center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            return image, avg_pupil_size

    return image, None


# Streamlit app
st.set_page_config(page_title="Facial Emotion Detection", page_icon="🙂", layout="centered")
st.title("Facial Emotion Detection with Webcam")
st.markdown("This application uses a webcam feed to detect emotions in real-time with a pre-trained deep learning model.")

tab1, tab2 = st.tabs(["Camera Tab", "Settings Tab"])


with tab2:
    # Sidebar for controls
    st.header("Settings")
    st.write("Adjust additional settings below:")
    brightness = st.slider("Brightness", 0, 100, 50)
    contrast = st.slider("Contrast", 0, 100, 50)
    zoom_level = st.slider("Zoom Level", 1, 10, 1)

with tab1:
    data_log = {"Timestamp": [], "Pupil Size": []}
    FRAME_WINDOW = st.empty()
    camera = None
    chart_placeholder = st.empty()

    if "run" not in st.session_state:
        st.session_state["run"] = False

        if 'data_shown' not in st.session_state:
            st.session_state['data_shown'] = False

        if 'data_log' not in st.session_state:
            st.session_state['data_log'] = {"Timestamp": [], "Pupil Size": []}

    # Buttons for starting/stopping the webcam
    start_button = st.button("Start Webcam")
    stop_button = st.button("Stop Webcam")

    if start_button:
        with st.spinner("Starting Webcam..."):
            st.session_state["run"] = True
            st.session_state['data_shown'] = False
            camera = cv2.VideoCapture(0)
            time.sleep(2)  # Wait for the camera to initialize

    if stop_button:
        st.session_state["run"] = False
        if camera is not None and camera.isOpened():
            camera.release()

    if st.session_state["run"]:
        while st.session_state["run"] and camera is not None and camera.isOpened():
            ret, frame = camera.read()
            if not ret:
                st.error("Unable to access the webcam. Please check your device.")
                break

            # Adjust brightness and contrast
            frame = cv2.convertScaleAbs(frame, alpha=contrast / 50, beta=brightness - 50)

            # Apply zoom
            h, w, _ = frame.shape
            center_x, center_y = w // 2, h // 2
            radius_x, radius_y = w // (2 * zoom_level), h // (2 * zoom_level)
            cropped_frame = frame[center_y - radius_y:center_y + radius_y, center_x - radius_x:center_x + radius_x]
            frame = cv2.resize(cropped_frame, (w, h))

            # Apply pupil detection logic
            processed_frame, pupil_size = detect_pupil(frame)

            # Log data
            st.session_state['data_log']["Timestamp"].append(time.strftime("%Y-%m-%d %H:%M:%S"))
            st.session_state['data_log']["Pupil Size"].append(pupil_size)

            # Add a border around the webcam display
            border_color = (255, 0, 0)  # Red border
            border_thickness = 10
            frame_with_border = cv2.copyMakeBorder(
                    processed_frame,
                    top=border_thickness,
                    bottom=border_thickness,
                    left=border_thickness,
                    right=border_thickness,
                    borderType=cv2.BORDER_CONSTANT,
                    value=border_color,
            )

            # Convert the frame to RGB (for Streamlit)
            frame_rgb = cv2.cvtColor(frame_with_border, cv2.COLOR_BGR2RGB)

            faces = faceDetect.detectMultiScale(frame_rgb, 1.3, 3)

            for x, y, w, h in faces:
                sub_face_img = frame_rgb[y:y + h, x:x + w]
                resized = cv2.resize(sub_face_img, (48, 48))
                grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                normalize = grayscale / 255.0  # Normalize pixel values to the range [0, 1]
                reshaped = np.reshape(normalize, (1, 48, 48, 1))  # Reshape for the model
                result = model.predict(reshaped)
                label = np.argmax(result, axis=1)[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
                cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
                cv2.putText(frame, labels[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Webcam Feed", use_column_width=True)

    if not st.session_state["run"]:
        st.info("Webcam is turned off. Use the buttons above to control it.")



