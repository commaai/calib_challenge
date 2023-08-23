import cv2
import numpy as np
import matplotlib.pyplot as plt

def getIntersection(line_equations):

    # Define the line equations in the form Ax + By = C for multiple lines
    # Each row represents a line equation [A, B, C]


    # Stack coefficients A, B into a matrix A
    A = line_equations[:, :2].astype(np.float64)

    # Stack coefficient C into a vector B
    B = line_equations[:, 2:].astype(np.float64)

    # Solve the linear system using np.linalg.lstsq
    intersection_point = np.linalg.lstsq(A, B, rcond=None)[0]

    return intersection_point

def getRANSACIntersection(line_equations, ransac_iterations=100, ransac_residual_threshold=0.01):
    num_lines, _ = line_equations.shape
    #print(num_lines)

    # Initialize variables to keep track of the best intersection point
    best_inliers = []
    best_intersection_point = None

    for _ in range(ransac_iterations):
        # Randomly select two lines for potential intersection
        sample_indices = np.random.choice(num_lines, 2, replace=False)
        sample_lines = line_equations[sample_indices]

        # Stack coefficients A, B into a matrix A
        A = sample_lines[:, :2].astype(np.float64)

        # Stack coefficient C into a vector B
        B = sample_lines[:, 2:].astype(np.float64)

        # Calculate intersection point using linear least squares
        intersection_point = np.linalg.lstsq(A, B, rcond=None)[0]

        # Calculate residuals (distances) of all lines to the intersection point
        residuals = np.abs(np.dot(line_equations[:, :2], intersection_point) - line_equations[:, 2])

        # Count inliers (lines with residuals smaller than threshold)
        inliers = np.where(residuals < ransac_residual_threshold)[0]

        # Update best intersection point if this iteration has more inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_intersection_point = intersection_point

    return best_intersection_point


def plot_lines_and_intersection(lines, intersection):
    plt.figure(figsize=(8, 8))
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)
    
    # Plot lines
    for line in lines:
        A, B, C = line
        if B != 0:
            x = np.linspace(-10, 10, 2)
            y = (C - A * x) / B
            plt.plot(x, y, label=f'{A:.2f}x + {B:.2f}y = {C:.2f}')
        else:
            plt.axvline(x=C/A, label=f'{A:.2f}x + {B:.2f}y = {C:.2f}')
    
    # Plot intersection point
    plt.plot(intersection[0], intersection[1], 'ro', label='Intersection Point')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Lines and Intersection Point')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':

    line_equations = np.array([
        [2, -1, 100],    # Example line equation 1
        [-3, 1, -50],    # Example line equation 2
        [1, 1, 1],       # Example line equation 3
        # ... add more line equations here
    ])
    # Print the intersection point
    intersection_point = getIntersection(line_equations)
    print("Intersection Point (x, y):", intersection_point)

    plot_lines_and_intersection(line_equations, intersection_point)
