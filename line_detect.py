import cv2
import numpy as np
import argparse

from contrast_test import increaseContrast
from line_intersection import getIntersection, plot_lines_and_intersection

MAX_LINE_SLOPE = 1.5
MIN_LINE_SLOPE = 0.1
MIN_LINE_LENGTH = 80

SENSOR_WIDTH = 1164
SENSOR_HEIGHT = 874

ACCEPTANCE_BOX_LOWER_LEFT = [(SENSOR_HEIGHT/2) - SENSOR_HEIGHT * 0.1, (SENSOR_WIDTH/2) - SENSOR_WIDTH * 0.1]
ACCEPTANCE_BOX_UPPER_RIGHT = [(SENSOR_HEIGHT/2) + SENSOR_HEIGHT * 0.1, (SENSOR_WIDTH/2) + SENSOR_WIDTH * 0.1]

def slopeOfLine(x1, y1, x2, y2):
    return (y2-y1)/(x2-x1)

def lengthOfLine(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2) 

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


def getLines(image):

    # Convert image to grayscale

    image = region_selection(image)

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #contrast = increaseContrast(gray)
    
    # Use canny edge detection
    edges = cv2.Canny(gray,50,150,apertureSize=3)
    
    # Apply HoughLinesP method to
    # to directly obtain line end points
    
    lines = cv2.HoughLinesP(
                edges, # Input edge image
                1, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians
                threshold=30, # Min number of votes for valid line
                minLineLength=5, # Min allowed length of line
                maxLineGap=100 # Max allowed gap between line for joining them
                )
    
    #print(lines)

    lines_list =[]

    # Iterate over points
    for points in lines:
        # Extracted points nested in the list
        x1,y1,x2,y2=points[0]

        lineSlope = slopeOfLine(x1, y1, x2, y2)
        lineLength = lengthOfLine(x1, y1, x2, y2)



        if(abs(lineSlope) < MAX_LINE_SLOPE and abs(lineSlope) > MIN_LINE_SLOPE and lineLength > MIN_LINE_LENGTH):

            #print("Line coordinates: ", x1, y1, x2, y2)
            #print("Line slope: ", lineSlope)
            #print("Line length: ", lineLength)

            b = y1 - lineSlope * x1
            m = lineSlope
            n = 1

            lineEquation = [m, n, b]

            #if(checkRegion(lineEquation)):

                # Draw the lines joing the points
                # On the original image
            cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
            # Maintain a simples lookup list for points
            lines_list.append([(x1,y1),(x2,y2), lineSlope])
        
  
    # Save the result image

    return image, lines_list
    #return image, lines_list

def region_selection(image):
	"""
	Determine and cut the region of interest in the input image.
	Parameters:
		image: we pass here the output from canny where we have
		identified edges in the frame
	"""
	# create an array of the same size as of the input image
	mask = np.zeros_like(image)
	# if you pass an image with more then one channel
	if len(image.shape) > 2:
		channel_count = image.shape[2]
		ignore_mask_color = (255,) * channel_count
	# our image only has one channel so it will go under "else"
	else:
		# color of the mask polygon (white)
		ignore_mask_color = 255
	# creating a polygon to focus only on the road in the picture
	# we have created this polygon in accordance to how the camera was placed
	rows, cols = image.shape[:2]
	bottom_left = [cols * 0.1, rows * 0.75]
	top_left	 = [cols * 0.1, rows * 0.5]
	bottom_right = [cols * 0.9, rows * 0.75]
	top_right = [cols * 0.9, rows * 0.5]
	vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
	# filling the polygon with white color and generating the final mask
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	# performing Bitwise AND on the input image and mask to get only the edges on the road
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image


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

    print("lines list: ", lines_list)
    print("Num Lines: ", len(lines_list))

    

    cv2.imshow('detectedLines.png',image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()