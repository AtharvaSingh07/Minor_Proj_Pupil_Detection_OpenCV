import cv2
import streamlit as st
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
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


# Streamlit app setup
st.set_page_config(page_title="Pupil Detection", page_icon="👁", layout="centered")
st.title("👁 Pupil Detection with Webcam")

# Sidebar for controls
st.sidebar.header("Settings")
st.sidebar.markdown("---")
st.sidebar.write("Adjust additional settings below:")
brightness = st.sidebar.slider("Brightness", 0, 100, 50)
contrast = st.sidebar.slider("Contrast", 0, 100, 50)
st.sidebar.markdown("---")
st.sidebar.write("Tested with Digital 10X Zoom Megapixel WebCam")

FRAME_WINDOW = st.empty()
camera = cv2.VideoCapture(0)

# Button to start/stop webcam
if 'run' not in st.session_state:
    st.session_state['run'] = False

start_button = st.button("Start Webcam")
stop_button = st.button("Stop Webcam")

if start_button:
    st.session_state['run'] = True
if stop_button:
    st.session_state['run'] = False

if st.session_state['run']:
    st.markdown(
        "<style>"
        "#MainMenu {visibility: hidden;}"
        "footer {visibility: hidden;}"
        "header {visibility: hidden;}"
        "</style>",
        unsafe_allow_html=True,
    )

    while st.session_state['run']:
        ret, frame = camera.read()
        if not ret:
            st.warning("Unable to access the webcam. Please check your device.")
            break

        # Adjust brightness and contrast
        frame = cv2.convertScaleAbs(frame, alpha=contrast/50, beta=brightness-50)

        # Apply pupil detection logic
        processed_frame, pupil_size = detect_pupil(frame)

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

        # Display the frame and pupil size
        FRAME_WINDOW.image(frame_rgb, caption="Webcam Feed", use_container_width=True)

    camera.release()
else:
    st.info("Webcam is turned off. Use the buttons above to control it.")
    camera.release()
