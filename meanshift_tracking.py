# Import Libraries
import cv2
import numpy as np


# Capture Video Stream
cap = cv2.VideoCapture("Video/face_track.mp4")
# Take first frame of the video
ret, frame = cap.read()
print(frame.shape)
# Set up the initial tracking window
face_casc = cv2.CascadeClassifier("Haarcascade/haarcascade_frontalface_default.xml")
face_rects = face_casc.detectMultiScale(frame)
# Convert the list to a tuple
face_x, face_y, w, h = tuple(face_rects[0])
track_window = (face_x, face_y, w, h)
# set up the ROI for tracking
roi = frame[face_y:face_y+h, face_x:face_x+w]
# HSV color maping
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# Histogram to target on each frame for the meanshift calculation
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0,180])
# Normalize the histogram
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX);


# Set the termination criteria
# 10 iterations or move 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


# While loop
while (cap.isOpened()):
    # capture video
    ret, frame = cap.read()    
    if ret:    
        # Frame in HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Calculate the base of ROI
        dest_roi = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # Meanshift to get the new coordinates of rectangle
        ret, track_window = cv2.meanShift(dest_roi, track_window, term_crit)
        # Draw new rectangle on image
        x, y, w, h = track_window
        # Open new window and display
        image2 = cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,255), 3)
        cv2.imshow("Face Tracker", image2)
        # Close window
        if cv2.waitKey(50) &0xFF == 27:
            break  
    else:
        break


# Release and Destroy
cap.release()
cv2.destroyAllWindows()