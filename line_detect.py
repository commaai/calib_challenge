import cv2
import numpy as np
import argparse

def absoluteSlopeOfLine(x1, y1, x2, y2):
    return abs((y2-y1)/(x2-x1))

def lengthOfLine(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2) 


def getLines(image):

    # Convert image to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    # Use canny edge detection
    edges = cv2.Canny(gray,50,150,apertureSize=3)
    
    # Apply HoughLinesP method to
    # to directly obtain line end points
    lines_list =[]
    lines = cv2.HoughLinesP(
                edges, # Input edge image
                1, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians
                threshold=100, # Min number of votes for valid line
                minLineLength=5, # Min allowed length of line
                maxLineGap=10 # Max allowed gap between line for joining them
                )

    # Iterate over points
    for points in lines:
        # Extracted points nested in the list
        x1,y1,x2,y2=points[0]

        lineSlope = absoluteSlopeOfLine(x1, y1, x2, y2)

        if(lineSlope < 1.5 and lineSlope > 0.375 and lengthOfLine(x1, y1, x2, y2) > 80):
            # Draw the lines joing the points
            # On the original image
            cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
            # Maintain a simples lookup list for points
            lines_list.append([(x1,y1),(x2,y2), lengthOfLine(x1, y1, x2, y2)])
        
  
    # Save the result image


    return image, lines_list


if __name__ == "__main__":
    # Create an argparse object to parse command-line arguments
    parser = argparse.ArgumentParser(description='Process an image using OpenCV.')
    parser.add_argument('image_path', type=str, help='Path to the input image file')
    args = parser.parse_args()

    # Read the image path from the command-line argument
    image_path = args.image_path

    # Read the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print('Error: Could not load the image.')
        exit()    
    
    image, lines_list = getLines(image)

    print(lines_list)

    cv2.imshow('detectedLines.png',image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()