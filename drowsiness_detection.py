import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from EAR_calculator import *
from imutils import face_utils
from imutils.video import VideoStream
import matplotlib.pyplot as plt
import matplotlib.animation as animate
from matplotlib import style
import imutils
import dlib
import time
import argparse
import cv2
from playsound import playsound
from scipy.spatial import distance as dist
import os
import csv
import numpy as np
import pandas as pd
from datetime import datetime
import threading

style.use('fivethirtyeight')

# Creating the dataset 
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Function to play sound in a separate thread
def play_sound(file):
    playsound(file)

# Function to compute the eye aspect ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to compute the mouth aspect ratio
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    mar = (A + B + C) / (2.0 * D)
    return mar

# All eye and mouth aspect ratio with time
ear_list = []
total_ear = []
mar_list = []
total_mar = []
ts = []
total_ts = []

# Construct the argument parser and parse the arguments 
ap = argparse.ArgumentParser() 
ap.add_argument("-p", "--shape_predictor", required=True, help="path to dlib's facial landmark predictor")
args = vars(ap.parse_args())

# Declare a constant which will work as the threshold for EAR value, below which it will be regarded as a blink 
EAR_THRESHOLD = 0.23
# Declare another constant to hold the consecutive number of frames to consider for a blink 
CONSECUTIVE_FRAMES = 20 
# Another constant which will work as a threshold for MAR value
MAR_THRESHOLD = 0.5

# Initialize two counters 
BLINK_COUNT = 0 
FRAME_COUNT = 0 

# Now, initialize the dlib's face detector model as 'detector' and the landmark predictor model as 'predictor'
print("[INFO] Loading the predictor...")
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor(args["shape_predictor"])

# Grab the indexes of the facial landmarks for the left and right eye respectively 
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Now start the video stream and allow the camera to warm-up
print("[INFO] Loading Camera...")
vs = VideoStream(src=0).start()
time.sleep(2) 

assure_path_exists("dataset/")
count_sleep = 0
count_yawn = 0 

# Now, loop over all the frames and detect the faces
while True: 
    # Extract a frame 
    frame = vs.read()
    cv2.putText(frame, "PRESS 'q' TO EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3) 
    # Resize the frame 
    frame = imutils.resize(frame, width=500)
    # Convert the frame to grayscale 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces 
    rects = detector(frame, 1)

    # Now loop over all the face detections and apply the predictor 
    for (i, rect) in enumerate(rects): 
        shape = predictor(gray, rect)
        # Convert it to a (68, 2) size numpy array 
        shape = face_utils.shape_to_np(shape)

        # Draw a rectangle over the detected face 
        (x, y, w, h) = face_utils.rect_to_bb(rect) 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)    
        # Put a number 
        cv2.putText(frame, "Driver", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        leftEye = shape[lstart:lend]
        rightEye = shape[rstart:rend] 
        mouth = shape[mstart:mend]
        # Compute the EAR for both the eyes 
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Take the average of both the EAR
        EAR = (leftEAR + rightEAR) / 2.0
        # Live data write in csv
        ear_list.append(EAR)
        
        ts.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
        # Compute the convex hull for both the eyes and then visualize it
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        # Draw the contours 
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)

        MAR = mouth_aspect_ratio(mouth)
        mar_list.append(MAR / 10)

        # Display EAR and MAR on the frame
        cv2.putText(frame, f"EAR: {EAR:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"MAR: {MAR:.2f}", (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Check if EAR < EAR_THRESHOLD, if so then it indicates that a blink is taking place 
        # Thus, count the number of frames for which the eye remains closed 
        if EAR < EAR_THRESHOLD: 
            FRAME_COUNT += 1

            cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

            if FRAME_COUNT >= CONSECUTIVE_FRAMES: 
                count_sleep += 1
                # Add the frame to the dataset as proof of drowsy driving
                cv2.imwrite("dataset/frame_sleep%d.jpg" % count_sleep, frame)
                threading.Thread(target=play_sound, args=('sound files/alarm.mp3',)).start()
                cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else: 
            if FRAME_COUNT >= CONSECUTIVE_FRAMES: 
                threading.Thread(target=play_sound, args=('sound files/warning.mp3',)).start()
            FRAME_COUNT = 0

        # Check if the person is yawning
        if MAR > MAR_THRESHOLD:
            count_yawn += 1
            cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1) 
            cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Add the frame to the dataset as proof of drowsy driving
            cv2.imwrite("dataset/frame_yawn%d.jpg" % count_yawn, frame)
            threading.Thread(target=play_sound, args=('sound files/alarm.mp3',)).start()
            threading.Thread(target=play_sound, args=('sound files/warning_yawn.mp3',)).start()

    # Total data collection for plotting
    for i in ear_list:
        total_ear.append(i)
    for i in mar_list:
        total_mar.append(i)            
    for i in ts:
        total_ts.append(i)
    # Display the frame 
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF 
    
    if key == ord('q'):
        break

# Save data to CSV
df = pd.DataFrame({"EAR": total_ear, "MAR": total_mar, "TIME": total_ts})
df.to_csv("op_webcam.csv", index=False)

# Plot EAR and MAR over time
df['TIME'] = pd.to_datetime(df['TIME'], format='%H:%M:%S.%f')
plt.figure(figsize=(10, 5))
plt.plot(df['TIME'], df['EAR'], label='EAR', color='blue', linewidth=2)
plt.plot(df['TIME'], df['MAR'], label='MAR', color='orange', linewidth=2)
plt.axhline(y=EAR_THRESHOLD, color='r', linestyle='--', label='EAR Threshold')
plt.axhline(y=MAR_THRESHOLD, color='g', linestyle='--', label='MAR Threshold')
plt.xlabel('Time')
plt.ylabel('Aspect Ratios')
plt.title('EAR & MAR over Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cleanup
cv2.destroyAllWindows()
vs.stop()
