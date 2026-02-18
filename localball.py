import cv2
from ultralytics import YOLO
import threading
import time

# researched online for this specific library
# to run the yolov8 model in python
# it's open source and very user-friendly for beginners,
# which is why I chose it
# it already has a bunch of pre-trained models to detect objects
# with refinement I can include my own pre-trained model
# to include the lids. for now, just testing out the code.

# yolo model to detect ball (test case)
model = YOLO('yolov8n.pt')  # upgrade to yolov8s.pt for better accuracy

# this set keeps track of every unique object ID the AI has detected
# Using a set means if, let's say object with #1 tag stays on screen, 
# it only gets counted once
counted_ids = set()

# this class runs the camera in a separate "lane" 
# so the AI doesn't make the video stutter
class VideoStream:
    def __init__(self, url):
        self.capture = cv2.VideoCapture(url)
        # force the buffer to 1 so we never see "old" frames
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.capture.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.capture.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.capture.release()

# droidcam url to connect via wifi
# format for any droidcam port: http://[your IP address goes here]:4747/video
droidcam_url = "http://192.168.0.37:4747/video"

# start the threaded stream to fix latency
vs = VideoStream(droidcam_url).start()
time.sleep(2.0) # warm up sensor

while True:
    # ret variable returns true/false (if camera is working)
    # frame = the actual image matrix from the camera
    ret, frame = vs.read()

    # if camera fails (ret is false), stop the code
    if not ret or frame is None:
        print("Camera input invalid. Restart DroidCam!!")
        break

    # run YOLO on full frame
    # classes=[32] ensures ONLY sports balls are detected (this includes basketballs)
    # quick class references
    # 0: person | 2: car | 14: bird | 15: cat | 16: dog
    # 32: ball | 39: bottle | 41: cup | 63: laptop | 67: phone
    # full list: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
    
    # using model.track with persist=True allows the AI to "remember" objects.
    results = model.track(
        frame,
        persist=True,   # Required for tracking IDs to stay consistent
        conf=0.6,       # slightly higher confidence to filter out noise
        iou=0.5,        # Adjusted for better tracking performance
        imgsz=640,      
        classes=[32],   # Switched back to 32 for your basketball test
        verbose=False   
    )

    # Check if any objects were actually detected and tracked
    if results[0].boxes.id is not None:
        # Extract the unique IDs as a list of integers
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        # Add each ID to our "memory" set
        for track_id in track_ids:
            counted_ids.add(track_id)

    # The total count is now the size of our memory set
    total_count = len(counted_ids)

    # Visualize current detections (boxes on screen right now)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # quick memo: filtering by size
        # basketballs are usually larger in frame than cricket balls.
        # we only count it if the box is at least 30 pixels wide/tall.
        if (x2 - x1) > 30: 
             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2) # Orange for basketball

    # display cumulative count (The 2 + 2 = 4 logic)
    cv2.rectangle(frame, (0, 0), (600, 60), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"TOTAL: {total_count} | Press 'R' to Reset",
        (10, 40),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (0, 255, 0), # Green text for the "all time" count
        2
    )

    # show the output window
    cv2.imshow("Persistent Basketball Counter", frame)

    # handle keyboard inputs
    key = cv2.waitKey(1) & 0xFF
    
    # press 'q' to quit
    if key == ord('q'):
        break
        
    # press 'r' to reset the unique ID memory and clear the count
    if key == ord('r'):
        counted_ids.clear()
        print("Counter has been reset!")

# release camera and close windows
# use the custom stop function for the thread
vs.stop()
cv2.destroyAllWindows()