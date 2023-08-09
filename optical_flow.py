import numpy as np
import cv2 as cv
import argparse, math, time
from line_detect import getLines

videoPath = "./labeled/2.hevc"
textPath = "./labeled/2.txt"

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
                pitchList.append(format(float(columns[0]), ".4g"))
                yawList.append(format(float(columns[1]), ".4g"))
            else:
                print(f"Ignored line: {line.strip()}")
    
    return pitchList, yawList

def addYawPitch(image, pitchList, yawList, frameNumber):
    x1 = 10
    x2 = 600

    y = 30

    formattedPitch = format(float(pitchList[frameNumber]), ".4g")
    cv.putText(image, formattedPitch, (x1, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    formattedYaw = format(float(yawList[frameNumber]), ".4g")
    cv.putText(image, formattedYaw, (x2, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv.putText(image, str(f"Frame: {frameNumber}"), (x2+350, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return image


if __name__ == '__main__':
    cap = cv.VideoCapture(videoPath)
    #cap = cv.VideoCapture(args.image)


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
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    pitchList, yawList = getYawPitch(textPath)
    print(len(pitchList))

    frameNumber = 0
    x1 = 10
    x2 = 600

    y = 30

    while(frameNumber < len(pitchList)):
        ret, frame = cap.read()

        if not ret:
            print('No frames grabbed!')
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if(frameNumber % 10 == 0):
            mask = np.zeros_like(old_frame)

        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
            #returns points that are tracked successfully

        startTime = time.time()

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            lineLength = math.sqrt((a-c)**2 + (b-d)**2)

            if(lineLength > 10):
                mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                #frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
                image = cv.add(frame, mask)

                image = addYawPitch(image, pitchList, yawList, frameNumber)
                image, lineLength = getLines(image)

                #cv.putText(img, pitchList[frameNumber], (x1, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #cv.putText(img, yawList[frameNumber], (x2, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #cv.putText(img, str(f"Frame: {frameNumber}"), (x2+350, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                
                k = cv.waitKey(30) & 0xff

                if k == 27:
                    break
                
                cv.imshow('frame', image)

        # Now update the previous frame and previous points

        endTime = time.time()

        if(endTime - startTime < 0.5):
            time.sleep(0.5 - (endTime - startTime))

        
        print(f"Frame: {frameNumber}")

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        frameNumber += 1

    cv.destroyAllWindows()