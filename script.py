import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import streamlit as st
import numpy as np
import base64

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

class GestureRecognizer:
    def __init__(self):
        self.base_options = python.BaseOptions(model_asset_path='model/gesture_recognizer.task')
        self.options = vision.GestureRecognizerOptions(base_options=self.base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        recognition_result = self.recognizer.recognize(mp_image)

        if recognition_result.gestures:
            top_gesture = recognition_result.gestures[0][0]
            hand_landmarks = recognition_result.hand_landmarks
            cv2.putText(frame, f"Gesture: {top_gesture.category_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            for hand_landmark in hand_landmarks:
                for point in hand_landmark:
                    cv2.circle(frame, (int(point.x * frame.shape[1]), int(point.y * frame.shape[0])), 5, (0, 255, 0), -1)
        
        return frame

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(image_file):
    bin_str = get_base64_of_bin_file(image_file)
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:image/png;base64,{bin_str}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

def main():
    st.set_page_config(layout="wide")
    st.image('images/mrm-norm.png', use_column_width=True)
    
    # Set the background image
    set_background('images/background-p.jpg')

    st.title("Mars Rover Manipal - AI Research")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Real-Time Gesture Recognition")
        st.write("This webapp shows live webcam feed with gesture recognition, try thumbs up, thumbs down, point up and victory!")
        
        # Initialize gesture recognizer
        recognizer = GestureRecognizer()
        
        # OpenCV VideoCapture
        cap = cv2.VideoCapture(0)
        
        # Streamlit image placeholder
        image_placeholder = st.empty()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam")
                break
            
            # Process the frame
            processed_frame = recognizer.process_frame(frame)
            
            # Convert BGR to RGB
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display the processed frame
            image_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)
    
    with col2:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")

if __name__ == "__main__":
    main()