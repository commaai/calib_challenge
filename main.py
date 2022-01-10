import numpy as np
import cv2
import time
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.ma.core import size

class Processing:
    def __init__(self):
        self.focalLen = 910
        
    def process_frame(self, img):
        height, width, z = img.shape
               
        img = cv2.resize(img, (width//2, height//2))
        
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(img, None)
        for p in kp:
            u,v = map(lambda x: int(round(x)), p.pt)
            cv2.circle(img, (u,v), color=(0,255,0), radius=1)
            # img2 = cv2.drawKeypoints(img, kp, None, color=(50, 255, 0))
        
        self.drawImg("Keypoints on Img", img, kp)
        
        return img, kp, des
    
    def matcher(self, img1, kp1, img2, kp2, des1, des2):
        #creates the brute force matcher
        #pass in NORM_HAMMING because ORB uses binary string descriptors
        
        #set crossCheck equal to True for better accuracy
        bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
        matches = bf.match(des1, des2)
        #best matches got to the front of the matches array
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Draw first 10 matches.
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:5], img1, flags=2)
        self.drawImg("Association", img3) 
            
    def drawImg(self, title ,frame, kp=None):
        cv2.imshow(title, frame)
        cv2.waitKey(1) #0xFF == ord('q')
        


if __name__ == "__main__":
    check = Processing()
    cap = cv2.VideoCapture('unlabeled/6.hevc')
    
    # fps= int(cap.get(cv2.CAP_PROP_FPS))
    # print("This is the fps ", fps)
    
    while cap.isOpened():
        prevFrame = None
        ret, currentFrame= cap.read()
        prevFrame = currentFrame
        if ret == True:
            img2, kp2, des2 = check.process_frame(currentFrame)
            img1 = img2
            kp1 = kp2
            des1 = des2
            # check.matcher(img1, kp1, img2, kp2, des1, des2)
        else:
            break