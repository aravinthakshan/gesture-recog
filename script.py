import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

class GestureRecognitionTransformer(VideoTransformerBase):
    def __init__(self):
        self.base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
        self.options = vision.GestureRecognizerOptions(base_options=self.base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        recognition_result = self.recognizer.recognize(mp_image)

        if recognition_result.gestures:
            top_gesture = recognition_result.gestures[0][0]
            hand_landmarks = recognition_result.hand_landmarks

            cv2.putText(img, f"Gesture: {top_gesture.category_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            for hand_landmark in hand_landmarks:
                for point in hand_landmark:
                    cv2.circle(img, (int(point.x * img.shape[1]), int(point.y * img.shape[0])), 5, (0, 255, 0), -1)

        return img

def main():
    st.set_page_config(layout="wide")
    def load_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    load_css("style.css")
    
  
    col1, col2= st.columns(2)

    with col1:
        st.title("Real-Time Gesture Recognition")
        st.write("This app shows live webcam feed with gesture recognition, try thumbs up, thumbs down, point up, victory !")
        webrtc_streamer(key="gesture-recognition", video_transformer_factory=GestureRecognitionTransformer)
        
        
    with col2:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.image("logo.png",width = 600,caption="By RESEARCH AI MRM")
        
if __name__ == "__main__":
    main()
