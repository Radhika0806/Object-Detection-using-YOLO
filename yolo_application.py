import streamlit as st 
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os


# Streamlit starts
st.title('Yolo image video processing')

# Giving an option to user for uploading it's files (adding uploading button)
uploaded_file = st.file_uploader("Upload a image or video file", type=['jpg', 'jpeg', 'png', 'webp', 'mp4', 'mkv'])

# loading YOLO model
model = YOLO("C:/Users/hp/Documents/Programming/Projects/ObjectDetection/best.pt")

# PROCESSING THE MEDIA
# checking file extensions and conforming if the file is correct or not
def process_media(input_path, output_path):
    file_extension = os.path.splitext(input_path)[1].lower()
    if file_extension in ['.mp4', '.mkv']:
        
        return predict_and_plot_video(input_path, output_path)
    elif file_extension in ['.jpg', '.jpeg', '.png', '.webp']:
        return predict_and_save_image(input_path, output_path)
    else:
        st.error(f"unsupported file type: {file_extension}")
        return None


# Performing predictions
def predict_and_save_image(path_test_car, output_image_path):
    results = model.predict(path_test_car, device = 'cpu')
    image = cv2.imread(path_test_car)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for result in results:
        for box in result.boxes:
           x1, y1, x2, y2 = map(int, box.xyxy[0])
           confidence = box.conf[0]
           cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
           cv2.putText(image, f'{confidence*100:.2f}%',
                       (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_path, image)
    return output_image_path


def predict_and_plot_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error openening video file: {video_path}")
        return None
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb_frame, device = 'cpu')
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{confidence*100:.2f}%',
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        out.write(frame)
    cap.release()
    out.release()
    return output_path 


# checking if the uploaded file is there or not
# now we have to save the input image and send it to the model as output
if uploaded_file is not None:
    input_path = f"temp/{uploaded_file.name}" #Saving data in temp directory
    output_path = f"output/{uploaded_file.name}"
    with open(input_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    st.write("Processing Image.....")
    result_path = process_media(input_path, output_path)
    # todo: Build logic for prediction
    if result_path:
        if input_path.endswith(('.mp4', '.mkv')):
            pass
            # st.video(result_path)
        else:
            st.image(result_path)


