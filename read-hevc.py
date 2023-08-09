import cv2, subprocess
from optical_flow import getYawPitch, displayYawPitch

textFilePath = "./labeled/0.txt"
yawList, pitchList = getYawPitch(textFilePath)

def play_hevc_file(file_path):
    try:
        vlc_path = r"/Applications/VLC.app/Contents/MacOS/VLC"  # Modify this path based on your installation
        subprocess.Popen([vlc_path, file_path])
        print(f"Playing HEVC file: {file_path}")
    except Exception as e:
        print(f"Error: {e}")


def extract_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    frame_count = 0
    
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        
        # Break the loop if the video is over or there's an issue reading frames
        if not ret:
            break
        
        # Process the frame here (you can perform inference or other tasks)
        # For demonstration, let's save each frame as an image
        frame_filename = f"frame_{frame_count:04d}.jpg"
        
        cv2.imwrite(frame_filename, frame)
        sample = cv2.imread(frame_filename)

        print(f"Extracted frame: {frame_filename}")
        displayYawPitch(sample, pitchList, yawList, frame_count)

        cv2.imshow("Image", sample)
        # Wait for the user to press a key
        #cv2.waitKey(0)
        # Close all windows
        #cv2.destroyAllWindows()
        
        frame_count += 1
        break
    
    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()


video_path = "./labeled/0.hevc"  # Replace with the actual path to your HEVC file
#play_hevc_file(hevc_file_path)

extract_frames(video_path)
