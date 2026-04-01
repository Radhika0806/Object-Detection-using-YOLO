import streamlit as st 
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os


# Setting up the Streamlit app title
st.title('YOLO Image and Video Processing Application')

# File uploader to allow users to upload images/videos
uploaded_file = st.file_uploader("Upload an image or video file", type=['jpg', 'jpeg', 'png', 'webp', 'mp4', 'mkv'])

# Loading the YOLO model (ensure 'best.pt' is the correct weights file)
model = YOLO(os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt"))

# Function to determine the type of media (image/video) and process accordingly
def process_media(input_path, output_path):
    # Extracting file extension
    file_extension = os.path.splitext(input_path)[1].lower()
    if file_extension in ['.mp4', '.mkv']:
        # If it's a video file, process it frame by frame
        return predict_and_plot_video(input_path, output_path)
    elif file_extension in ['.jpg', '.jpeg', '.png', '.webp']:
        # If it's an image file, process it to detect objects
        return predict_and_save_image(input_path, output_path)
    else:
        # Display an error message if an unsupported file type is uploaded
        st.error(f"Unsupported file type: {file_extension}")
        return None


# Function to predict and annotate objects in an image
def predict_and_save_image(path_test_car, output_image_path):
    try:
        # Perform model prediction
        results = model.predict(path_test_car, device='cpu')
        image = cv2.imread(path_test_car)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f'{confidence*100:.2f}%', (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path, image)
        return output_image_path
    except Exception as e:
        st.error(f"Error during image prediction: {e}")
        return None


# Function to process video frames for object detection
def predict_and_plot_video(video_path, output_path):
    try:
        # Capturing video frames for processing
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Error opening video file: {video_path}")
            return None

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Setting up the video writer for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                # Perform model prediction
                results = model.predict(rgb_frame, device='cpu')
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = box.conf[0]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{confidence*100:.2f}%', (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                st.error(f"Error during frame prediction: {e}")
            out.write(frame)
        cap.release()
        out.release()
        return output_path
    except Exception as e:
        st.error(f"Error during video processing: {e}")
        return None


# Main logic to handle uploaded files
if uploaded_file is not None:
    # Ensure the directories for temporary and output files exist
    input_path = f"temp/{uploaded_file.name}"  # Saving data in temp directory
    output_path = f"output/{uploaded_file.name}"
    os.makedirs(os.path.dirname(input_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Inform the user that their file is being processed
    st.write("Processing Image or Video...")
    result_path = process_media(input_path, output_path)

    # Display results
    if result_path:
        if input_path.endswith(('.mp4', '.mkv')):
            # Display processed video
            st.video(result_path)
        else:
            # Display processed image
            st.image(result_path)
