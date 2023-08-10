from __future__ import print_function
from builtins import input
import cv2 as cv
import numpy as np
import argparse
from tqdm import tqdm
# Initialize values

# Do the operation new_image(i,j) = alpha*image(i,j) + beta
# Instead of these 'for' loops we could have used simply:
# new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
# but we wanted to show you how to access the pixels :)

def increaseContrast(image):

    new_image = np.zeros(image.shape, image.dtype)

    alpha = 2.0 # Simple contrast control
    beta = 0 # Simple brightness control

    for y in tqdm(range(image.shape[0])):
        for x in range(image.shape[1]):
            #for c in range(image.shape[2]):
            new_image[y,x] = np.clip(alpha*image[y,x] + beta, 0, 255)
    
    return new_image


if __name__ == "__main__":
    # Read image given by user
    parser = argparse.ArgumentParser(description='Process an image using OpenCV.')
    parser.add_argument('image_path', type=str, help='Path to the input image file')
    args = parser.parse_args()

    # Read the image path from the command-line argument
    image_path = args.image_path

    # Read the image
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    print(image.shape)


    if image is None:
        print('Could not open or find the image: ', args.input)
        exit(0)

    cv.imshow('Original Image', image)

    new_image = increaseContrast(image)
    cv.imshow('New Image', new_image)
    # Wait until user press some key
    cv.waitKey()