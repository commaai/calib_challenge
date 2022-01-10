The comma.ai Calibration Challenge!
======

Goal is to predict the direction of travel (in camera frame) from provided dashcam video.

- This repo provides 10 videos. Every video is 1min long and 20 fps.
- 5 videos are labeled with a 2D array describing the direction of travel at every frame of the video with a pitch and yaw angle in radians.
- 5 videos are unlabeled. It is your task to generate the labels for them.
- The example labels are generated using a Neural Network, and the labels were confirmed with a SLAM algorithm (by Comma AI team).
- The focal length is asked to be assumed 910 pixels.


![picture](https://user-images.githubusercontent.com/6804392/116619874-e78a8180-a8f5-11eb-93e3-c9c852726db8.png)

Context for the challenge
------
The devices that run [openpilot](https://github.com/commaai/openpilot/) are not mounted perfectly. The camera is not exactly aligned to the vehicle. There is some pitch and yaw angle between the camera of the device and the vehicle, which can vary between installations Estimating these angles is essential for accurate control of the vehicle. The best way to start estimating these values is to predict the direction of motion in camera frame. More info  can be found in [this readme](https://github.com/commaai/openpilot/tree/master/common/transformations).

Deliverable
-----

We have to create 5 labels called 5.txt to 9.txt. These labels should be a 2D array that contains the pitch and yaw angles of the direction of travel (in camera frame) of every frame of the respective videos. Zip them up and e-mail it to givemeajob@comma.ai.

Evaluation
-----

We will evaluate your mean squared error against our ground truth labels. Errors for frames where the car speed is less than 4m/s will be ignored. Those are also labeled as NaN in the example labels.

This repo includes an eval script that will give an error score (lower is better). We can use it to test our solutions against the labeled examples.

TO DO:
-----

Score an error under 25% on the unlabeled set.
