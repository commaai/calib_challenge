import cv2
import os
import numpy as np
from tqdm import tqdm
import time

from optical_flow import getYawPitch, returnIntersection, coordinatesToGyro, export_data_to_txt, plotLineOnFrame


data = "0"

videoPath = os.path.join("./labeled/" + data + ".hevc")
textPath = os.path.join("./labeled/" + data + ".txt")

pitchList, yawList = getYawPitch(textPath)

# Open the video file
cap = cv2.VideoCapture(videoPath)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()


FRAME_WIDTH = int(cap.get(3))  # Width of the frames in the video
FRAME_HEIGHT = int(cap.get(4))  # Height of the frames in the video
NEW_WIDTH = int(FRAME_WIDTH/3)
NEW_HEIGHT = int(FRAME_HEIGHT/3)

# Create an output video file
fps = int(cap.get(5))
output_path = 'output_video_original.mp4'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (NEW_WIDTH, NEW_HEIGHT))

def imageResize(frame, new_width, new_height):
    resized = cv2.resize(frame, (new_width, new_height))
    return resized

def preprocessImage(image):
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
    edges = cv2.Canny(blurred_frame, threshold1=30, threshold2=100)

    return edges

def getDistance(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def getSlope(x1, y1, x2, y2):
    return (y2-y1)/(x2-x1)

def getLineMatrix(x, y, slope):
    b = -slope*x + y
    return ([-slope, 1, b])

def checkQuadrants(x1, y1, slope):
    y_divider = 2
    if((x1 < NEW_WIDTH/2 and y1 < NEW_HEIGHT/y_divider) or (x1 > NEW_WIDTH/2 and y1 > NEW_HEIGHT/y_divider)):
        if(slope > 0):
            return False
        
    elif((x1 > NEW_WIDTH/2 and y1 > NEW_HEIGHT/y_divider) or (x1 < NEW_WIDTH/2 and y1 < NEW_HEIGHT/y_divider)):
        if(slope < 0):
            return False
        
    return True
    


def opticalFlow(inferenceFrame, prevFrame, frame, feature_params, lk_params, color=(0, 255, 0)):

    mask = np.zeros_like(inferenceFrame)

    prev = cv2.goodFeaturesToTrack(prevFrame, mask = None, **feature_params)
    next, status, error = cv2.calcOpticalFlowPyrLK(prevFrame, inferenceFrame, prev, None, **lk_params)
    # Selects good feature points for previous position
    goodOld = prev[status == 1].astype(int)
    # Selects good feature points for next position
    goodNew = next[status == 1].astype(int)
    # Draws the optical flow tracks
    lines = []

    for i, (new, old) in enumerate(zip(goodNew, goodOld)):
        # Returns a contiguous flattened array as (x, y) coordinates for new point
        a, b = new.ravel()
        # Returns a contiguous flattened array as (x, y) coordinates for old point
        c, d = old.ravel()

        #if(getDistance(a, b, c, d) > 5 and checkQuadrants(a, b, getSlope(a, b, c, d))):
        # Draws line between new and old position with green color and 2 thickness
        
        # Draws filled circle (thickness of -1) at new position with green color and radius of 3
        #inferenceFrame = cv2.circle(inferenceFrame, (a, b), 3, (255,0,0), -1)

        slope = getSlope(a, b, c, d)
        
        if(0 < abs(slope) < 10000 and checkQuadrants(a, b, slope)):
        #if(0 < abs(slope) < 10000):
            lines.append(getLineMatrix(a, b, slope))
            mask = cv2.line(inferenceFrame, (a, b), (c, d), (255,0,0), 4)

    output = cv2.add(inferenceFrame, mask)

    intersection = returnIntersection(lines)

    output = cv2.circle(inferenceFrame, (int(intersection[0]), int(intersection[1])), 10, (255,0,0), -1)
    intersection

    return output, intersection


def getLines(frame):
    startTime = time.time()

    image, lines = getLines(frame)

    avgLineNum += len(lines)

    lineMatrix = []

    for line in lines:
        #Getting line equation in form mx + ny = b
        #Matrix in [m, n, b]
        line_equation = getLineMatrix(line[0][0], line[0][1], line[2])
        
        #if(checkRegion(line_equation)):
        lineMatrix.append(line_equation)
        frame = plotLineOnFrame(NEW_WIDTH/1.5, NEW_WIDTH, line_equation, frame)


def main():
    # Optical Flow Parameters
    feature_params = dict( maxCorners = 200,
                            qualityLevel = 0.2,
                            minDistance = 12,
                            blockSize = 10)

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize = (15, 15),
        maxLevel = 2,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    counter = 0
    yawInferenceList = []
    pitchInferenceList = []

    for index in tqdm(range(len(pitchList))):
        # Read a frame from the video
        ret, frame = cap.read()

        if not ret:
            break  # Break the loop if no more frames are available

        if(index == 0):
            prevFrame = imageResize(frame, NEW_WIDTH, NEW_HEIGHT)
            prevFrame = preprocessImage(prevFrame)

        # Preprocessing Steps
        # 1. Image Resizing (Optional)
        frame = imageResize(frame, NEW_WIDTH, NEW_HEIGHT)
        inferenceFrame = preprocessImage(frame)

        # You can add more preprocessing steps as needed

        # Write the preprocessed frame to the output video
        #out.write(edges)  # Change 'edges' to the preprocessed frame you want to save

        # --- Optical Flow --- 
        output, intersection = opticalFlow(inferenceFrame, prevFrame, frame, feature_params, lk_params, counter)
        yawInference, pitchInference = coordinatesToGyro(intersection[0], intersection[1])

        yawInferenceList.append(yawInference)
        pitchInferenceList.append(pitchInference)

        cv2.imshow("Original Frame", frame)
        cv2.imshow("Optical Flow", inferenceFrame)

        colourImage = np.stack((inferenceFrame,) * 3, axis=-1)

        out.write(frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        counter+=1
        old_gray = prevFrame.copy()

    filename = "./data/" + data + "_" + str(time.time()) + ".txt"

    # Call the function to export data to the text file
    #export_data_to_txt(yawInferenceList, pitchInferenceList, filename)

    # Release video objects and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #main()

    ret, frame = cap.read()

    
