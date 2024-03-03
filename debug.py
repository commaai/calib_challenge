import cv2
import numpy as np

def boundary_filter(image, top_left, bottom_right):
    # Create a mask of the same size as the input image
    mask = np.zeros_like(image, dtype=np.uint8)

    # Define the region of interest (ROI) based on the top-left and bottom-right coordinates
    roi = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # Fill the corresponding region in the mask with white (255)
    mask[roi] = 255

    # Apply bitwise AND operation to the input image and the mask
    filtered_image = cv2.bitwise_and(image, mask)

    return filtered_image

# Load an example image
image = cv2.imread('frame_0000.jpg')

# Define the top-left and bottom-right coordinates of the boundary
top_left = (100, 100)
bottom_right = (400, 300)

# Apply the boundary filter to the image
filtered_image = boundary_filter(image, top_left, bottom_right)

# Display the original and filtered images
cv2.imshow('Original Image', image)
cv2.imshow('Filtered Image', filtered_image)

# Wait for a key press
cv2.waitKey(0)