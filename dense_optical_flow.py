# Import Libraries
import cv2
import numpy as np


# Video Capture
cap = cv2.VideoCapture('Video/chaplin.mp4')
# Read the capture and get the first frame
ret, first_frame = cap.read()
# Convert frame to Grayscale
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
# Create# Create an image with the same dimensions as the frame for later drawing purposes
mask = np.zeros_like(first_frame)
# Saturation to maximum
mask[..., 1] = 255


# While loop
while(cap.isOpened()):
    if ret:
        # Read the capture and get the first frame
        ret, frame = cap.read()
        # Open new window and display the input frame
        cv2.imshow('input', frame)
        # Convert all frame to Grayscale (previously we did only the first frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate dense optical flow by Farneback
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Compute Magnitude and Angle
        magn, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Set image hue depanding on the optical flow direction
        mask[..., 0] = angle*180/np.pi/2
        # Normalize the magnitude
        mask[..., 2] = cv2.normalize(magn, None, 0, 255, cv2.NORM_MINMAX)
        # Convert HSV to RGB
        rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB)
        # Open new window and display the output
        cv2.imshow("Dense Optical Flow", rgb)
        # Update previous frame
        prev_gray = gray
        # Close the frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

    
# Release and Destroy
cap.release()
cv2.destroyAllWindows()


