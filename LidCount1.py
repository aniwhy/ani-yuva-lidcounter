import streamlit as st
import cv2
from ultralytics import YOLO
import os
import numpy as np

# researched online for this specific library 
# to run the yolov8 model in python
# it's open source and very user-friendly for beginners,
# which is why I chose it
# it already has a bunch of pre-trained models to detect objects
# with refinement I can include my own pre-trained model
# to include the lids. for now, just testing out the code.

# yolo model to detect ball (test case)
model = YOLO('yolov8n.pt') 

# create a folder on the laptop to save photos
# this will be used later to train the AI on actual lids
if not os.path.exists('lid_images'):
    os.makedirs('lid_images')

# set up the professional dashboard layout
# wide mode makes it easier to see on a factory tablet
st.set_page_config(page_title="All-Clad Inventory", layout="wide")
st.title("📊 All-Clad: Lid Counter Prototype")

# columns to separate the camera from the data/buttons
col1, col2 = st.columns([3, 1])

with col2:
    st.header("Inventory Metrics")
    # big number for clear visibility from a distance
    count_placeholder = st.empty()
    st.write("---")
    # quick memo: streamlit buttons for interaction
    save_btn = st.button("📸 Save Image for Training")
    st.info("System Status: Monitoring Tote")

with col1:
    # placeholder to push frames to the web page
    image_placeholder = st.empty()

# droidcam url to connect via wifi
# format for any droidcam port: http://[your IP address goes here]:4747/video
droidcam_url = "http://192.168.0.37:4747/video"
capture = cv2.VideoCapture(droidcam_url)

# counter to give each saved image a unique name
if 'img_counter' not in st.session_state:
    st.session_state.img_counter = 0

while True:
    # quick memo: 
    # ret variable returns true/false (if camera is working)
    # frame = the actual image matrix from the camera
    ret, frame = capture.read()
    
    # if camera fails (ret is false), stop the code
    if not ret:
        st.error("Camera input invalid. Restart DroidCam!!")
        break

    # variables to define the bucket area (roi)
    # used to focus on the tote as per problem 1
    h, w, _ = frame.shape
    bx1, by1 = int(w * 0.25), int(h * 0.25)
    bx2, by2 = int(w * 0.75), int(h * 0.75)
    
    # crop frame to focus only on the bucket area
    bucket_area = frame[by1:by2, bx1:bx2]

    # run detection on bucket area only
    # conf=0.5 means AI must be 50% sure it's a ball
    # this prevents from wacky fluctuations in counting 
    # when the model is unsure about an object
    results = model(bucket_area, conf=0.5)
    
    # filter for sports ball, which is class ID 32 in the dataset
    # this stops background objects from being counted
    ball_boxes = [box for box in results[0].boxes if int(box.cls) == 32]
    count = len(ball_boxes)

    # quick memo: press 's' or click button to save a photo
    # this fulfills the note about getting images from multiple angles
    if save_btn:
        img_name = f"lid_images/test_shot_{st.session_state.img_counter}.png"
        cv2.imwrite(img_name, frame)
        st.toast(f"Saved: {img_name}")
        st.session_state.img_counter += 1
        save_btn = False # resets button state

    # draw the blue boundary for the bucket zone
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
    
    # draw green boxes around detected balls
    for box in ball_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # offset coordinates back to main frame so boxes line up
        cv2.rectangle(frame, (x1 + bx1, y1 + by1), (x2 + bx1, y2 + by1), (0, 255, 0), 2)

    # convert BGR to RGB so colors look right in the browser
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # update the web elements live
    image_placeholder.image(frame_rgb, channels="RGB")
    count_placeholder.metric("BALLS IN BUCKET", count)

# release camera when done
capture.release()
