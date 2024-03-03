import cv2
import os
import numpy as np
from tqdm import tqdm
import time
import argparse

from optical_flow import getYawPitch, returnIntersection, coordinatesToGyro, export_data_to_txt, plotLineOnFrame

def main():
    parser = argparse.ArgumentParser(description='Process an image using OpenCV.')
    parser.add_argument('data', type=str, help='Path to the data file set')
    args = parser.parse_args()

    data = args.data
    videoPath = os.path.join("./labeled/" + data + ".hevc")
    textPath = os.path.join("./labeled/" + data + ".txt")

    pitchList, yawList = getYawPitch(textPath)

    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        exit()

    FRAME_WIDTH = int(cap.get(3))
    FRAME_HEIGHT = int(cap.get(4))
    NEW_WIDTH = int(FRAME_WIDTH/3)
    NEW_HEIGHT = int(FRAME_HEIGHT/3)

    fps = int(cap.get(5))
    output_path = 'output_video_original.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (NEW_WIDTH, NEW_HEIGHT))

    # Params for optical flow
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    color = (0, 255, 0)

    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    mask = np.zeros_like(old_frame)

    intersectionList = []
    yawInferenceList = []
    pitchInferenceList = []

    for index in tqdm(range(len(pitchList))):
        frame = pitchList[index]
        ret, frame = cap.read() 

        if not ret:
            print('No frames grabbed!')
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lineMatrix = []

        # Optical Flow Method
        output, lineMatrix = opticalFlow(frame_gray, old_gray, frame, feature_params, lk_params, color)
        
        intersection = returnIntersection(lineMatrix)
        output = cv2.circle(frame, (int(intersection[0]), int(intersection[1])), 10, color, -1)
        intersectionList.append(intersection)

        yawInference, pitchInference = coordinatesToGyro(intersection[0], intersection[1])
        yawInferenceList.append(yawInference)
        pitchInferenceList.append(pitchInference)

        cv2.imshow("output", output)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cv2.destroyAllWindows()

    verticalFilename = "./data/pitch/" + data + "_" + str(time.time()) + ".txt"
    horizontalFilename = "./data/yaw/" + data + "_" + str(time.time()) + ".txt"
    filename = "./data/" + args.data + "_" + str(time.time()) + ".txt"

    export_data_to_txt(yawInferenceList, pitchInferenceList, filename)

if __name__ == '__main__':
    main()
