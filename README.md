# VisionProjectPPP

VisionProjectPPP is a Python-based application that I made to fulfill the requirements for my college subject "Applied Programming Practicum". It combines computer vision, gesture recognition, and object detection. It uses Kivy for the graphical user interface, MediaPipe for gesture recognition, and YOLO for object detection. The project also includes a text-to-speech (TTS) system and a database for storing detected objects.

## Features

- **Gesture Recognition**: Detects hand gestures such as "Thumbs Up", "Thumbs Down", and "Closed Fist" using MediaPipe.
- **Object Detection**: Identifies objects in real-time using the YOLOv8 model.
- **Zoom Functionality**: Allows zooming into detected objects for better visualization.
- **Text-to-Speech (TTS)**: Provides audio feedback for detected gestures and objects using Google Text-to-Speech (gTTS).
- **Database Integration**: Stores detected objects with metadata (e.g., confidence, bounding box, timestamp) in an SQLite database.
- **Camera Selection**: Supports multiple camera inputs with a dropdown for easy switching.
- **Interactive GUI**: Built with Kivy, featuring gesture feedback, camera feed, and database management.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/VisionProjectPPP.git
   cd VisionProjectPPP
    ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
3. Ensure you have a camera connected to your system.

## Usage

1. Run the application:
   ```bash
   python main.py
   ```

2. Use gestures to interact with the application:
   - **Thumbs Up**: Start object detection.
   - **Thumbs Down**: Pause detection and capture objects.
   - **Closed Fist**: Reset the system.

3. Use the GUI buttons:
   - **Camera**: Switch between available cameras.
   - **Database**: View and manage detected objects.

## Project Structure

- `main.py`: Entry point for the application.
- `Camera.py`: Core logic for camera feed, gesture recognition, and object detection.
- `TTS.py`: Text-to-speech manager for audio feedback.
- `database.py`: SQLite database handler for storing detected objects.
- `GUI.py`: Kivy-based graphical user interface components.
- `README.md`: Project documentation.

## Requirements

- Python 3.8 or higher
- OpenCV
- MediaPipe
- Kivy
- YOLOv8 (via `ultralytics` package)
- gTTS
- SQLite3

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push them to your fork.
4. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for gesture recognition.
- [YOLO](https://github.com/ultralytics/yolov5) for object detection.
- [Kivy](https://kivy.org/) for the GUI framework.
- [gTTS](https://pypi.org/project/gTTS/) for text-to-speech functionality.

## Contact

For questions or feedback, please contact [postmaster@sarifindustries.org].
