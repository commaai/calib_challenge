import cv2
import os
import numpy as np

from optical_flow import optical_flow

data = "./labeled/3"

videoPath = os.path.join(data + ".hevc")

# Open the video file
cap = cv2.VideoCapture(videoPath)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Create an output video file
#output_path = 'output_video.avi'
frame_width = int(cap.get(3))  # Width of the frames in the video
frame_height = int(cap.get(4))  # Height of the frames in the video
#out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))
new_width = int(frame_width/3)
new_height = int(frame_height/3)

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if no more frames are available

    # Preprocessing Steps
    # 1. Image Resizing (Optional)
    frame = cv2.resize(frame, (new_width, new_height))

    # 2. Grayscale Conversion
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. Noise Reduction (Gaussian Blur)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # 4. Edge Detection (Canny)
    edges = cv2.Canny(blurred_frame, threshold1=30, threshold2=100)

    # You can add more preprocessing steps as needed

    # Write the preprocessed frame to the output video
    #out.write(edges)  # Change 'edges' to the preprocessed frame you want to save

    # Display the original and preprocessed frames (for visualization purposes)
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Preprocessed Frame", edges)  # Change 'edges' to the preprocessed frame you want to display

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video objects and close windows
cap.release()
#out.release()
cv2.destroyAllWindows()
