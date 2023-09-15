import numpy as np
import cv2 as cv
import argparse, math, time
from line_detect import getLines
from line_detect2 import frame_processor, region_selection, canny_edge_detector

from line_intersection import getIntersection, plot_lines_and_intersection, getRANSACIntersection
#from eval_local import evalAccuracy

import os, math
from tqdm import tqdm


#Following measurements are in pixels
FOCAL_LENGTH = 910

#Sensor dimensions taken from image pixel dimensions
SENSOR_WIDTH = 1164
SENSOR_HEIGHT = 874

HORIZONTAL_FOV = 2 * math.atan((SENSOR_WIDTH / 2) / FOCAL_LENGTH) * 180 / math.pi
VERTICAL_FOV = 2 * math.atan((SENSOR_HEIGHT / 2) / FOCAL_LENGTH) * 180 / math.pi
print(HORIZONTAL_FOV)
print(VERTICAL_FOV)

ACCEPTABLE_FRAME_FRACTION = 0.1

ACCEPTANCE_BOX_LOWER_LEFT = [(SENSOR_HEIGHT/2) - SENSOR_HEIGHT * 0.1, (SENSOR_WIDTH/2) - SENSOR_WIDTH * 0.1]
ACCEPTANCE_BOX_UPPER_RIGHT = [(SENSOR_HEIGHT/2) + SENSOR_HEIGHT * 0.1, (SENSOR_WIDTH/2) + SENSOR_WIDTH * 0.1]

def getYawPitch(file_path):

    pitchList = []
    yawList = []

    with open(file_path, 'r') as file:
    # Read each line of the file
        for line in file:
            # Split the line into columns using whitespace as the delimiter
            columns = line.strip().split()
            #strip = remove whitespace
            #split = split string into list
            
            # Check if the line has exactly two columns

            if len(columns) == 2:
                pitchList.append(format(float(columns[0]) * 180/math.pi, ".4g") )
                yawList.append(format(float(columns[1]) * 180/math.pi, ".4g") )
            else:
                print(f"Ignored line: {line.strip()}")
    
    return pitchList, yawList

def addYawPitch(image, pitchList, yawList, frameNumber):
    x1 = 10
    x2 = 600

    y = 30

    formattedPitch = "Pitch: " + format(float(pitchList[frameNumber]), ".4g")
    cv.putText(image, formattedPitch, (x1, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    formattedYaw = "Yaw: " + format(float(yawList[frameNumber]), ".4g")
    cv.putText(image, formattedYaw, (x2, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv.putText(image, str(f"Frame: {frameNumber}"), (x2+350, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return image


def getLineMatrix(x, y, slope):
    b = -slope*x + y
    return ([-slope, 1, b])


def export_data_to_txt(yawList, pitchList, filename):
    with open(filename, 'w') as file:
        for frame in tqdm(range(len(yawList))):
            
            line = str(yawList[frame]) + " " + str(pitchList[frame]) + "\n"
    
            file.write(line)


def checkQuadrants(line_equation):
    print(line_equation)

    YMin = SENSOR_HEIGHT/ACCEPTABLE_FRAME_FRACTION
    YMax = (ACCEPTABLE_FRAME_FRACTION-1)*YMin

    XMin = SENSOR_WIDTH/ACCEPTABLE_FRAME_FRACTION
    XMax = (ACCEPTABLE_FRAME_FRACTION-1)*XMin

    points = []
    points.append([0, YMin, 0, YMax])
    points.append([SENSOR_WIDTH, YMin, SENSOR_WIDTH, YMax])

    points.append([XMin, 0, XMax, 0])
    points.append([XMin, SENSOR_HEIGHT, XMax, SENSOR_HEIGHT])

    checkRegion(line_equation, points)

    return False

def checkRegion(line_equation, lowerLeft=ACCEPTANCE_BOX_LOWER_LEFT, upperRight=ACCEPTANCE_BOX_UPPER_RIGHT):

    m = line_equation[0]
    n = line_equation[1]
    b = line_equation[2]
    #print(m, n, b)

    lowerLeftX = b + lowerLeft[1] * m   
    #print(lowerLeftX, lowerLeft[0], upperRight[0])  
    if(lowerLeft[0] <= lowerLeftX <= upperRight[0]):
        return True
    
    upperRightX = b - upperRight[1] * m
    if(upperRight[0] <= upperRightX <= upperRight[0]):
        return True
    
    lowerLeftY = b - lowerLeft[0] * n
    if(lowerLeft[1] <= lowerLeftY <= upperRight[1]):
        return True
    
    upperRightY = b - upperRight[0] * n
    if(upperRight[1] <= upperRightY <= upperRight[1]):
        return True
    
    return False


import numpy as np
import matplotlib.pyplot as plt

def plot_lines_from_matrices(line_matrices, x_range, num_points):
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    
    for matrix in line_matrices:
        slope, _, b = matrix
        y_values = slope * x_values + b
        plt.plot(x_values, y_values, label=f'Slope: {slope:.2f}, b: {b:.2f}')

    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Lines from Line Matrices')
    plt.legend()
    plt.grid(True)
    plt.show()

def plotLineOnFrame(x1, x2, lineEquation, frame):

    m = lineEquation[0]
    y1 = m*x1 + lineEquation[2]
    y2 = m*x2 + lineEquation[2]

    frame = cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    return frame

def opticalFlow(gray, prev_gray, frame, feature_params, lk_params, color):

    mask = np.zeros_like(frame)
    lines = []

    prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
    next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
    # Selects good feature points for previous position
    good_old = prev[status == 1].astype(int)
    # Selects good feature points for next position
    good_new = next[status == 1].astype(int)
    # Draws the optical flow tracks

    lineCounter = 0

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # Returns a contiguous flattened array as (x, y) coordinates for new point (a, b)
        x2, y2 = new.ravel()
        # Returns a contiguous flattened array as (x, y) coordinates for old point (c, d)
        x1, y1 = old.ravel()
        # Draws line between new and old position with green color and 2 thickness
        
        # Draws filled circle (thickness of -1) at new position with green color and radius of 3
        #frame = cv.circle(frame, (a, b), 3, color, -1)

        coordinates = []
        coordinates.append([x2, y2])
        coordinates.append([x1, y1])

        slope = (y2-y1)/(x2-x1)
        if(0 < abs(slope) < 10000):
            lines.append(getLineMatrix(x2, y2, slope))

            print(lines[lineCounter])

            # Showing lines for visualization
            y1 = SENSOR_HEIGHT
            x1 = int(y1-lines[lineCounter][2]/lines[lineCounter][0])

            if(x1 > 0 and x1 < SENSOR_WIDTH):
                coordinates[1] = [x1, y1]

            y2 = 0
            x2 = int(y2-lines[lineCounter][2]/lines[lineCounter][0])
            
            if(x2 > 0 and x2 < SENSOR_WIDTH):
                coordinates[0] = [x2, y2]


            if(x1 < 0 or x2 > SENSOR_WIDTH):
                x1 = 0
                y1 = int(lines[lineCounter][2])
                if(y1 > 0 and y1 < SENSOR_HEIGHT):
                    coordinates[1] = [x1, y1]

                x2 = SENSOR_WIDTH
                y2 = int(x2*lines[lineCounter][0] + lines[lineCounter][2])                
                
                if(y2 > 0 and y2 < SENSOR_HEIGHT):
                    coordinates[0] = [x2, y2]



            print('coordinates: ', coordinates)
            
            lineCounter+=1

            x1 = coordinates[0][0]
            y1 = coordinates[0][1]
            x2 = coordinates[1][0]
            y2 = coordinates[1][1]
        
            mask = cv.line(frame, (x1, y1), (x2, y2), color, 3)

        #print("iteration: ", i)
    # Overlays the optical flow tracks on the original frame

    #print("frame: ", frame.shape)
    #print("mask: ", mask.shape)

    output = cv.add(frame, mask)
    # Updates previous frame
    prev_gray = gray.copy()
    # Updates previous good feature points
    prev = good_new.reshape(-1, 1, 2)
    # Opens a new window and displays the output frame
    #cv.imshow("sparse optical flow", output)

    return output, lines



def main():
    
    parser = argparse.ArgumentParser(description='Process an image using OpenCV.')
    parser.add_argument('data', type=str, help='Path to the data file set')
    args = parser.parse_args()

    # Read the image path from the command-line argument

    data = "./labeled/" + args.data

    videoPath = os.path.join(data + ".hevc")
    textPath = os.path.join(data + ".txt")

    cap = cv.VideoCapture(videoPath)
    #cap = cv.VideoCapture(args.image)

#   --- Params for optical flow

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )


    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize = (15, 15),
    maxLevel = 2,
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = (0, 255, 0)

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
#   ---

    pitchList, yawList = getYawPitch(textPath)
    print("total frames: ", len(pitchList))

    frameNumber = 0
    x1 = 10
    x2 = 600

    y = 30

    intersectionList = []
    verticalPixelDeviationList = []
    horizontalPixelDeviationList = []

    yawInferenceList = []
    pitchInferenceList = []

    avgLineNum = 0

    #while(frameNumber < len(pitchList)):
    for frame in tqdm(range(len(pitchList))):

        ret, frame = cap.read() 

        if not ret:
            print('No frames grabbed!')
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #print("frame", frame)
        #print("gray", frame_gray)

        lineMatrix = []
        #--- Optical Flow Method ---
        output, lineMatrix = opticalFlow(frame_gray, old_gray, frame, feature_params, lk_params, color)
        #print(lines)

        #plot_lines_from_matrices(lines, (0, 1000), len(lines))


        

        #--- Line Detect Method ---
        '''
        startTime = time.time()

        image, lines = getLines(frame)

        avgLineNum += len(lines)


        for line in lines:
            #Getting line equation in form mx + ny = b
            #Matrix in [m, n, b]
            line_equation = getLineMatrix(line[0][0], line[0][1], line[2])
            
            #if(checkRegion(line_equation)):
            lineMatrix.append(line_equation)
            frame = plotLineOnFrame(SENSOR_WIDTH/1.5, SENSOR_WIDTH, line_equation, frame)
        '''

        #print(lineMatrix)

        #print("End of test")
        #break
        #print(lines)
        

        lineMatrix = np.array(lineMatrix)
        #print("line matrix: ", lineMatrix)
        #print("size: ", len(lineMatrix))

        intersection = None

        if(len(lineMatrix) >= 2):           
            #intersection = getRANSACIntersection(lineMatrix)
            intersection = getIntersection(lineMatrix)
            #print("intersection: ", intersection)

        if(intersection is None):
            intersection = [SENSOR_WIDTH/2, SENSOR_HEIGHT/2]
            #Return center of frame if no lines detected      

        #print("intersection: ", intersection)

        if(intersection[0] < 300):
            plot_lines_and_intersection(lineMatrix, intersection)
            
        #print("intersection: ", (int(intersection[0]), int(intersection[1])))
        output = cv.circle(frame, (int(intersection[0]), int(intersection[1])), 10, color, -1)
        intersectionList.append(intersection)

        horizontalPixelDeviation = abs(intersection[0] - SENSOR_WIDTH/2)
        verticalPixelDeviation = abs(intersection[1] - SENSOR_HEIGHT/2)

        yawInference = math.atan(horizontalPixelDeviation / FOCAL_LENGTH)
        pitchInference = math.atan(verticalPixelDeviation / FOCAL_LENGTH)

        horizontalPixelDeviationList.append(horizontalPixelDeviation)
        verticalPixelDeviationList.append(verticalPixelDeviation)

        yawInferenceList.append(yawInference)
        pitchInferenceList.append(pitchInference)

        #image = addYawPitch(image, pitchList, yawList, frameNumber) 
        #cv.imshow('frame', image)

        cv.imshow("output", output)
        cv.waitKey(0)

        k = cv.waitKey(30) & 0xff

        if k == 27:
            break

        # Now update the previous frame and previous points

        endTime = time.time()

        #if(endTime - startTime < 0.5):
            #time.sleep(0.5 - (endTime - startTime))

        
        #print(f"Frame: {frameNumber}")

        frameNumber += 1

    avgLineNum = avgLineNum/len(pitchList)
    print("----------------------", avgLineNum)

    cv.destroyAllWindows()

    verticalFilename = "./data/pitch/" + data + "_" + str(time.time()) + ".txt"
    horizontalFilename = "./data/yaw/" + data + "_" + str(time.time()) + ".txt"
    filename = "./data/" +args.data + "_" + str(time.time()) + ".txt"

    # Call the function to export data to the text file
    export_data_to_txt(yawInferenceList, pitchInferenceList, filename)

    #yawError, pitchError = evalAccuracy(yawList, pitchList, yawInferenceList, pitchInferenceList)
    #print("Yaw Error: ", yawError)
    #print("Pitch Error: ", pitchError)


if __name__ == '__main__':
    main()   
