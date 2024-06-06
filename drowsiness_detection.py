import datetime as dt
import matplotlib.pyplot as plt
from imutils import face_utils 
import matplotlib.animation as animation
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

# All eye and mouth aspect ratio with time
ear_list = []
total_ear = []
mar_list = []
total_mar = []
ts = []
total_ts = []

# Construct the argument parser and parse the arguments 
ap = argparse.ArgumentParser() 
ap.add_argument("-p", "--shape_predictor", required=True, help="Path to dlib's facial landmark predictor")
args = vars(ap.parse_args())

# Declare constants
EAR_THRESHOLD = 0.3
CONSECUTIVE_FRAMES = 20
MAR_THRESHOLD = 0.6  # Adjusted threshold for better detection

# Initialize counters
BLINK_COUNT = 0 
FRAME_COUNT = 0 

# Initialize dlib's face detector and facial landmark predictor
print("[INFO] Loading the predictor...")
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor(args["shape_predictor"])

# Grab the indexes of the facial landmarks for the left and right eye respectively 
(lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Start the video stream
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)
# time.sleep(2)

assure_path_exists("history/")
count_sleep = 0
count_yawn = 0 

# Initialize variables for alarm timing
last_alarm_time = 0
alarm_interval = 5  # Interval between alarms in seconds

# Function to play sound
def play_alarm(sound_file):
    playsound(sound_file)

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

# Loop over the frames from the video stream
while True: 
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    cv2.putText(frame, "PRESS 'q' TO EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3) 
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects): 
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect) 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)    
        cv2.putText(frame, "Driver", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        leftEye = shape[lstart:lend]
        rightEye = shape[rstart:rend] 
        mouth = shape[mstart:mend]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        EAR = (leftEAR + rightEAR) / 2.0
        ear_list.append(EAR)
        ts.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

        MAR = mouth_aspect_ratio(mouth)
        mar_list.append(MAR)

        current_time = time.time()

        if EAR < EAR_THRESHOLD: 
            FRAME_COUNT += 1
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)
            if FRAME_COUNT >= CONSECUTIVE_FRAMES: 
                count_sleep += 1
                cv2.imwrite("history/frame_sleep%d.jpg" % count_sleep, frame)
                if current_time - last_alarm_time >= alarm_interval:
                    threading.Thread(target=play_alarm, args=('sound files/alarm.mp3',)).start()
                    last_alarm_time = current_time
                cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if FRAME_COUNT == 50: 
                    FRAME_COUNT = 0
        else: 
            FRAME_COUNT = 0

        if MAR > MAR_THRESHOLD:
            count_yawn += 1
            cv2.drawContours(frame, [mouthHull], -1, (0, 0, 255), 1)
            cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite("history/frame_yawn%d.jpg" % count_yawn, frame)

            if current_time - last_alarm_time >= alarm_interval:
                threading.Thread(target=play_alarm, args=('sound files/warning_yawn.mp3',)).start()
                last_alarm_time = current_time

    total_ear.extend(ear_list)
    total_mar.extend(mar_list)            
    total_ts.extend(ts)
    
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
cap.release()
