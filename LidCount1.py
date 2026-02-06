import cv2
from ultralytics import YOLO
import os

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

# droidcam url to connect via wifi
# format for any droidcam port: http://[your IP address goes here]:4747/video
droidcam_url = "http://192.168.0.37:4747/video"
capture = cv2.VideoCapture(droidcam_url)

# counter to give each saved image a unique name
img_counter = 0

while True:
    # quick memo: 
    # ret variable returns true/false (if camera is working)
    # frame = the actual image matrix from the camera
    ret, frame = capture.read()
    
    # if camera fails (ret is false), stop the code
    if not ret:
        print("Camera input invalid. Restart DroidCam!!")
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

    # draw the blue boundary for the bucket zone
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
    
    # draw green boxes around detected balls
    for box in ball_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # offset coordinates back to main frame so boxes line up
        cv2.rectangle(frame, (x1 + bx1, y1 + by1), (x2 + bx1, y2 + by1), (0, 255, 0), 2)

    # display the count of the objects on screen
    # added a small instruction line for the user
    cv2.rectangle(frame, (0, 0), (450, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"BALLS IN BUCKET: {count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "S: Save Image | Q: Quit", (10, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # show the final output window
    cv2.imshow("ball detection in bucket, problem 1 prototype 1", frame)

    # handle key presses
    key = cv2.waitKey(1)
    
    # quick memo: press 's' to save a photo
    # this fulfills the note about getting images from multiple angles
    if key & 0xFF == ord('s'):
        img_name = f"lid_images/test_shot_{img_counter}.png"
        cv2.imwrite(img_name, frame)
        print(f"Saved: {img_name}")
        img_counter += 1

    # press 'q' to quit the window
    elif key & 0xFF == ord('q'):
        break

# release camera and close windows to clean up memory
capture.release()
cv2.destroyAllWindows()