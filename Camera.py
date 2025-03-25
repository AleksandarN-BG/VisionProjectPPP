import os
import time
from types import SimpleNamespace

import cv2
import mediapipe as mp
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from mediapipe.tasks import python
from ultralytics import YOLO

from GUI import CameraView
from TTS import TTSManager

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

current_dir = os.path.dirname(os.path.abspath(__file__))
gesture_model_path = os.path.join(current_dir, 'gesture_recognizer.task')
latest_result = None
latest_objects = []
active_gesture = None
last_gesture_time = 0

# Read gesture model
with open(gesture_model_path, 'rb') as f:
    gesture_model_data = f.read()


# Global TTS instance
tts_manager = TTSManager()


def play_audio_sequence(sound_file=None, tts_text=None):
    """Queue sound file and TTS text to play in sequence"""
    try:
        if sound_file:
            tts_manager.play_file(sound_file)
        if tts_text:
            tts_manager.speak(tts_text)
    except Exception as e:
        print(f"Audio playback error: {e}")

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result, active_gesture, last_gesture_time

    # Debounce gestures (300ms minimum time between gestures)
    current_time = time.time()
    if current_time - last_gesture_time < 0.3:
        return

    # Gesture-specific confidence thresholds
    thresholds = {
        "Thumb_Up": 0.6,
        "Thumb_Down": 0.5,
        "Victory": 0.35,
        "Closed_Fist": 0.6,
        "Open_Palm": 0.6,
        "Pointing_Up": 0.6
    }
    default_threshold = 0.6

    if result.gestures and result.gestures[0]:
        # Get the top gesture
        gesture = result.gestures[0][0]
        gesture_name = gesture.category_name
        confidence = gesture.score

        # Use gesture-specific threshold or default
        threshold = thresholds.get(gesture_name, default_threshold)

        # Log all gesture detections for debugging
        print(f'Gesture detected: {gesture_name} with confidence {confidence:.2f} (threshold: {threshold:.2f})')

        if confidence > threshold and gesture_name != 'None':
            active_gesture = gesture_name
            latest_result = gesture_name
            last_gesture_time = current_time  # Update debounce timestamp
            print(f'*** ACCEPTED gesture: {gesture_name} with confidence {confidence:.2f} ***')
        else:
            print(f'Rejected gesture: {gesture_name} (confidence {confidence:.2f} below threshold {threshold:.2f})')

class KivyCameraApp(App):
    def rescale_frame_to_window(self):
        """Rescale the camera frame to fit the window size"""
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                h, w = frame.shape[:2]
                aspect_ratio = w / h
                new_width = int(Window.height * aspect_ratio)
                cv2.resize(frame, (new_width, Window.height), frame)

    def build(self):
        self.layout = CameraView()

        # Camera selection setup
        self.available_cameras = self.get_available_cameras()
        self.current_camera_index = 0  # Default camera
        self.setup_camera_dropdown()

        # Initialize camera
        self.cap = cv2.VideoCapture(self.current_camera_index)
        self.cap.set(3, 1920)
        self.cap.set(4, 1080)
        self.rescale_frame_to_window()

        # Application state variables
        self.active_gesture_state = None
        self.gesture_timestamp = 0
        self.last_timestamp = 0
        self.last_detected_gesture = None
        self.gesture_confidence_counter = 0
        self.captured_frame = None
        self.captured_objects = []
        self.zoom_object_index = 0
        self.zoom_level = 1.0
        self.welcome_played = False
        self.processing_gesture = False  # Lock for gesture processing

        # Initialize MediaPipe Hands for gesture detection
        try:
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
        except Exception as e:
            print(f"Error initializing MediaPipe Hands: {e}")

        # Initialize gesture recognizer
        self.recognizer = GestureRecognizer.create_from_options(gesture_options)

        # Initialize YOLO for object detection
        self.detector = YOLO("yolov8n.pt")  # Using YOLOv8 nano model

        # Schedule the frame updates
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)
        return self.layout

    def get_available_cameras(self):
        """Get list of available cameras"""
        try:
            # Simplified to avoid dependency
            return ["Default Camera"]
        except Exception as e:
            print(f"Error getting camera list: {e}")
            return ["Default Camera"]  # Fallback

    def setup_camera_dropdown(self):
        """Setup the camera selection dropdown"""
        dropdown = DropDown()

        for index, name in enumerate(self.available_cameras):
            btn = Button(
                text=f"{index}: {name}",
                size_hint_y=None,
                height=44
            )
            btn.bind(on_release=lambda btn, idx=index: self.select_camera(idx, dropdown))
            dropdown.add_widget(btn)

        # Bind the dropdown to the camera button
        self.layout.cam_button.bind(on_release=dropdown.open)

    def select_camera(self, index, dropdown):
        """Switch to the selected camera"""
        if index == self.current_camera_index:
            dropdown.dismiss()
            return

        # Release current camera
        if self.cap and self.cap.isOpened():
            self.cap.release()

        # Update camera index
        self.current_camera_index = index

        # Open new camera
        self.cap = cv2.VideoCapture(index)

        # Check if opened successfully
        if not self.cap.isOpened():
            print(f"Failed to open camera {index}")
            # Try to reopen the previous camera
            self.cap = cv2.VideoCapture(0)

        dropdown.dismiss()

    def process_yolo_results(self, results):
        """Process YOLO results to match the expected format for the app"""
        processed_objects = []

        try:
            if not results or len(results) == 0:
                return processed_objects

            # Get the first result (single image)
            result = results[0]

            # Process detections
            for detection in result.boxes:
                # Skip low confidence detections
                confidence = float(detection.conf)
                if confidence < 0.25:  # Confidence threshold
                    continue

                # Get box coordinates (convert tensor to int)
                box = detection.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                width = x2 - x1
                height = y2 - y1

                # Get class index and name
                cls_id = int(detection.cls[0])
                class_name = result.names[cls_id]

                # Create SimpleNamespace object to match expected format
                obj = SimpleNamespace()
                obj.bounding_box = SimpleNamespace(
                    origin_x=x1,
                    origin_y=y1,
                    width=width,
                    height=height
                )
                obj.categories = [
                    SimpleNamespace(
                        category_name=class_name,
                        score=confidence
                    )
                ]

                processed_objects.append(obj)

        except Exception as e:
            print(f"Error processing YOLO results: {e}")

        return processed_objects

    def update_frame(self, dt):
        global latest_result, latest_objects, active_gesture

        current_time = time.time()
        timestamp_ms = int(current_time * 1000)

        # If we're in PAUSED state, use the captured frame
        if self.active_gesture_state == "PAUSED":
            frame = self.captured_frame.copy()

            # Process gesture recognition on the live feed (even when display is paused)
            ret, live_frame = self.cap.read()
            if ret:
                # Process with MediaPipe Hands
                rgb_frame = cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB)
                hand_results = self.hands.process(rgb_frame)

                if hand_results.multi_hand_landmarks:
                    # Hand detection - isolate hand region
                    h, w, c = live_frame.shape
                    x_min, y_min = w, h
                    x_max, y_max = 0, 0

                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        for landmark in hand_landmarks.landmark:
                            x, y = int(landmark.x * w), int(landmark.y * h)
                            x_min = min(x_min, x)
                            y_min = min(y_min, y)
                            x_max = max(x_max, x)
                            y_max = max(y_max, y)

                    # Add padding
                    padding = 50
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)

                    # Extract hand region
                    hand_region = rgb_frame[y_min:y_max, x_min:x_max]

                    if hand_region.size > 0 and len(hand_region.shape) == 3 and hand_region.shape[2] == 3:
                        try:
                            hand_region_copy = np.ascontiguousarray(hand_region)
                            mp_hand_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=hand_region_copy)

                            if timestamp_ms > self.last_timestamp:
                                self.recognizer.recognize_async(mp_hand_image, timestamp_ms)
                                self.last_timestamp = timestamp_ms
                        except Exception as e:
                            print(f"Error creating MediaPipe image: {e}")
                else:
                    # No hands detected, use full frame
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                    if timestamp_ms > self.last_timestamp:
                        self.recognizer.recognize_async(mp_image, timestamp_ms)
                        self.last_timestamp = timestamp_ms

                # Check for Closed_Fist to reset
                if latest_result == "Closed_Fist" and not self.processing_gesture:
                    self.processing_gesture = True

                    if self.last_detected_gesture == "Closed_Fist":
                        self.gesture_confidence_counter += 1
                    else:
                        self.gesture_confidence_counter = 1
                        self.last_detected_gesture = "Closed_Fist"

                    if self.gesture_confidence_counter >= 3:
                        message = "Closed fist detected. Resetting system..."
                        self.active_gesture_state = None
                        self.layout.update_gesture_text(message)
                        self.zoom_level = 1.0
                        self.zoom_object_index = 0
                        Clock.schedule_once(lambda dt: self.reset_system(), 1.5)
                        play_audio_sequence(tts_text="Resetting system.")

                        # Display the message on the captured frame
                        cv2.putText(frame, message, (frame.shape[1] // 4, frame.shape[0] // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    self.processing_gesture = False
                elif latest_result != "Closed_Fist":
                    self.gesture_confidence_counter = 0
                    self.last_detected_gesture = latest_result
                    self.processing_gesture = False

            # Handle object zooming
            if self.captured_objects:
                if self.zoom_object_index < len(self.captured_objects):
                    obj = self.captured_objects[self.zoom_object_index]
                    bbox = obj.bounding_box
                    obj_name = obj.categories[0].category_name

                    # Zoom into the object
                    frame = self.zoom_into_object(frame, bbox, self.zoom_level)

                    # Label the object
                    label_text = f"Object: {obj_name}"
                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                    text_x = (frame.shape[1] - text_size[0]) // 2
                    text_y = frame.shape[0] - 20
                    cv2.putText(frame, label_text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                else:
                    message = "No non-person objects detected. Make closed fist to reset."
                    self.layout.update_gesture_text(message)
            else:
                message = "No objects to zoom in on. Make closed fist to reset."
                self.layout.update_gesture_text(message)

        # If not in PAUSED state, capture new frames
        else:
            ret, frame = self.cap.read()
            if not ret:
                return

            # Convert frame to RGB for processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe Hands
            hand_results = self.hands.process(rgb_frame)

            # Run YOLO object detection
            try:
                yolo_results = self.detector(rgb_frame, verbose=False)  # Add verbose=False to reduce output
                latest_objects = self.process_yolo_results(yolo_results)
            except Exception as e:
                print(f"Error running YOLO detection: {e}")
                latest_objects = []

            # Draw hand landmarks
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Get hand bounding box
                    h, w, c = frame.shape
                    x_min, y_min = w, h
                    x_max, y_max = 0, 0

                    for landmark in hand_landmarks.landmark:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                        x_max = max(x_max, x)
                        y_max = max(y_max, y)

                    # Add padding
                    padding = 50
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)

                    # Draw rectangle around hand region
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 105, 180), 2)

                    # Extract hand region for gesture recognition
                    hand_region = rgb_frame[y_min:y_max, x_min:x_max]

                    if hand_region.size > 0 and len(hand_region.shape) == 3 and hand_region.shape[2] == 3:
                        if timestamp_ms > self.last_timestamp:
                            hand_region_copy = np.ascontiguousarray(hand_region, dtype=np.uint8)
                            try:
                                mp_hand_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=hand_region_copy)
                                self.recognizer.recognize_async(mp_hand_image, timestamp_ms)
                                self.last_timestamp = timestamp_ms
                                cv2.putText(frame, "Hand region isolated",
                                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.7, (255, 105, 180), 2)
                            except TypeError as e:
                                print(f"Error creating MediaPipe image: {e}")
            else:
                # No hands detected, use full frame for gesture recognition
                if timestamp_ms > self.last_timestamp:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    self.recognizer.recognize_async(mp_image, timestamp_ms)
                    self.last_timestamp = timestamp_ms
                    cv2.putText(frame, "No hands detected - using full frame",
                                (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)

            # Debug information
            debug_info = f"Current gesture: {latest_result}, State: {self.active_gesture_state}"
            cv2.putText(frame, debug_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # State machine logic
            if self.active_gesture_state == "DETECTING":
                # Check for Thumbs down to capture
                if latest_result == "Thumb_Down" and not self.processing_gesture:
                    self.processing_gesture = True

                    # Capture and pause
                    self.captured_frame = frame.copy()
                    self.captured_objects = [obj for obj in latest_objects if
                                             obj.categories[0].category_name != "person"]

                    if self.captured_objects:
                        message = f"Thumbs down detected! Pausing on {len(self.captured_objects)} objects. Make closed fist to reset."
                        self.active_gesture_state = "PAUSED"
                        play_audio_sequence((os.path.join(os.path.dirname(os.path.abspath(__file__)), "gotobject.wav")), f"Object: {self.get_latest_objects()} detected.")
                    else:
                        message = "Thumbs down detected, but no objects found. Still detecting."

                    self.layout.update_gesture_text(message)
                    self.processing_gesture = False
                else:
                    # Still in DETECTING state, waiting for Thumbs down
                    message = "Detection active - make Thumbs down to capture object"

                # If thumbs up detected again
                if latest_result == "Thumb_Up":
                    message = "Already in detection mode - make Thumbs down to capture object"

                self.layout.update_gesture_text(message)

            # If not in DETECTING state, check for Thumb_Up
            else:
                if latest_result == "Thumb_Up" and not self.processing_gesture:
                    self.processing_gesture = True

                    if self.last_detected_gesture == "Thumb_Up":
                        self.gesture_confidence_counter += 1
                    else:
                        self.gesture_confidence_counter = 1
                        self.last_detected_gesture = "Thumb_Up"

                    # Enter DETECTING state after confirming Thumb_Up
                    if self.gesture_confidence_counter >= 3:
                        message = "Thumbs up detected! Object detection active. Make a thumbs down to capture."
                        self.active_gesture_state = "DETECTING"
                        self.gesture_timestamp = current_time
                        play_audio_sequence((os.path.join(os.path.dirname(os.path.abspath(__file__)), 'start.wav')),
                                            "Ready to go. Make a thumbs down gesture to capture objects.")
                    else:
                        message = "Keep holding thumbs up..."

                    self.processing_gesture = False
                elif latest_result != "Thumb_Up":
                    # Reset confidence counter
                    self.gesture_confidence_counter = 0
                    self.last_detected_gesture = latest_result

                    # Default message
                    message = "Make a thumbs up gesture to start object detection"
                    if not self.welcome_played:
                        play_audio_sequence(tts_text="Welcome! Make a thumbs up gesture to start object detection.")
                        self.welcome_played = True

                self.layout.update_gesture_text(message)

            # Draw bounding boxes for YOLO-detected objects
            for obj in latest_objects:
                bbox = obj.bounding_box
                cv2.rectangle(frame, (bbox.origin_x, bbox.origin_y),
                              (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                              (0, 255, 0), 2)
                label = f"{obj.categories[0].category_name}: {obj.categories[0].score:.2f}"
                cv2.putText(frame, label, (bbox.origin_x, bbox.origin_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display current gesture
            if active_gesture:
                cv2.putText(frame, f"Gesture: {active_gesture}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            # Display current state
            cv2.putText(frame, f"State: {self.active_gesture_state} (conf: {self.gesture_confidence_counter})",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Update the display
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.layout.img.texture = texture

    def zoom_into_object(self, frame, bbox, zoom_level):
        """Zoom into an object's bounding box"""
        h, w = frame.shape[:2]

        # Calculate center
        center_x = bbox.origin_x + bbox.width // 2
        center_y = bbox.origin_y + bbox.height // 2

        # Calculate zoom window size
        zoom_width = max(int(bbox.width * 3.0 / zoom_level), w // 4)
        zoom_height = max(int(bbox.height * 3.0 / zoom_level), h // 4)

        # Calculate cropping bounds
        x1 = max(0, center_x - zoom_width // 2)
        y1 = max(0, center_y - zoom_height // 2)
        x2 = min(w, center_x + zoom_width // 2)
        y2 = min(h, center_y + zoom_height // 2)

        # Ensure minimum size
        if x2 - x1 < w // 4:
            x2 = min(w, x1 + w // 4)
        if y2 - y1 < h // 4:
            y2 = min(h, y1 + h // 4)

        # Create zoomed frame
        zoomed = frame[y1:y2, x1:x2].copy()
        zoomed = cv2.resize(zoomed, (w, h))

        # Draw highlight rectangle
        rel_x = int((bbox.origin_x - x1) * w / (x2 - x1))
        rel_y = int((bbox.origin_y - y1) * h / (y2 - y1))
        rel_w = int(bbox.width * w / (x2 - x1))
        rel_h = int(bbox.height * h / (y2 - y1))

        cv2.rectangle(zoomed, (rel_x, rel_y), (rel_x + rel_w, rel_y + rel_h), (0, 0, 255), 3)

        return zoomed

    def reset_system(self):
        """Reset the system to its initial state"""
        self.active_gesture_state = None
        self.captured_frame = None
        self.captured_objects = []
        self.gesture_confidence_counter = 0
        self.last_detected_gesture = None
        self.layout.update_gesture_text("System reset. Make a thumbs up gesture to start object detection")

    def get_latest_objects(self):
        """Get names of detected objects"""
        object_list = []
        for obj in latest_objects:
            if obj.categories[0].category_name != "person":
                object_list.append(obj.categories[0].category_name)

        return ", ".join(object_list) if object_list else "nothing"

    def on_stop(self):
        """Clean up resources when app is closed"""
        self.cap.release()
        self.recognizer.close()
        self.hands.close()
        # No need to close YOLO model


gesture_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_buffer=gesture_model_data),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
