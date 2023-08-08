import cv2

# Specify the path to the text file
file_path = "./labeled/0.txt"

# Initialize lists to store the data from the two columns
pitch = []
yaw = []

# Open the file in read mode
with open(file_path, 'r') as file:
    # Read each line of the file
    for line in file:
        # Split the line into columns using whitespace as the delimiter
        columns = line.strip().split()
        #strip = remove whitespace
        #split = split string into list
        
        # Check if the line has exactly two columns
        if len(columns) == 2:
            pitch.append(columns[0])
            yaw.append(columns[1])
        else:
            print(f"Ignored line: {line.strip()}")

# Print the stored data from the two columns
print("Column 1 data:", pitch)
print("Column 2 data:", yaw)




imagePath = "frame_0000.jpg"
image = cv2.imread(imagePath)

x = 10
y = 30

cv2.putText(image, pitch[0], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

x = 600
y = 30

cv2.putText(image, yaw[0], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Image", image)

cv2.waitKey(0)
