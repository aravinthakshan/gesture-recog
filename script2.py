import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe's gesture recognizer
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Define the callback function to process and display the results
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    if result.gestures:
        top_gesture = result.gestures[0][0]
        print(f'Gesture: {top_gesture.category_name}, Confidence: {top_gesture.score}')
    else:
        print('No gesture recognized.')

    # Draw hand landmarks on the image
    if result.hand_landmarks:
        for hand_landmark in result.hand_landmarks:
            for landmark in hand_landmark:
                cv2.circle(output_image.numpy_view(), (int(landmark.x * output_image.width), int(landmark.y * output_image.height)), 5, (0, 255, 0), -1)
    
    # Display the image with the recognized gestures and hand skeleton
    cv2.imshow('Gesture Recognition', output_image.numpy_view())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False

# Load the model
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='/path/to/model.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# Initialize the recognizer
with GestureRecognizer.create_from_options(options) as recognizer:
    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the frame to the correct format for MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Get the current timestamp in milliseconds
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Perform gesture recognition asynchronously
        recognizer.recognize_async(mp_image, frame_timestamp_ms)

    cap.release()
    cv2.destroyAllWindows()
