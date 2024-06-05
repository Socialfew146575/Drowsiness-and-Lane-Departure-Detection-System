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
MAR_THRESHOLD = 14

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

assure_path_exists("dataset/")
count_sleep = 0
count_yawn = 0 

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
        mar_list.append(MAR / 10)

        if EAR < EAR_THRESHOLD: 
            FRAME_COUNT += 1
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)
            if FRAME_COUNT >= CONSECUTIVE_FRAMES: 
                count_sleep += 1
                cv2.imwrite("dataset/frame_sleep%d.jpg" % count_sleep, frame)
                playsound('sound files/alarm.mp3')
                cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if FRAME_COUNT == 50: 
                    FRAME_COUNT = 0
        else: 
            FRAME_COUNT = 0
        
        alarm_limit = 3  # Set the number of times you want the alarm to play
        alarm_count = 0  # Initialize a counter for the alarm

        if MAR > MAR_THRESHOLD:
            count_yawn += 1
            cv2.drawContours(frame, [mouthHull], -1, (0, 0, 255), 1)
            cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite("dataset/frame_yawn%d.jpg" % count_yawn, frame)

            if alarm_count < alarm_limit:  # Check if the alarm limit has been reached
                playsound('sound files/alarm.mp3')
                playsound('sound files/warning_yawn.mp3')
                alarm_count += 1  # Increment the alarm counter
            

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
df.plot(x='TIME', y=['EAR', 'MAR'])
plt.title('EAR & MAR calculation over time')
plt.ylabel('EAR & MAR')
plt.gca().axes.get_xaxis().set_visible(False)
plt.show()

# Cleanup
cv2.destroyAllWindows()
cap.release()
