Welcome to the comma.ai Calibration Challenge!
======

Your goal is to predict the direction of travel (in camera frame) from dashcam video.

- This repo provides 10 videos. Every video is 1min long and 20 fps.
- 5 videos are labeled with a 2D array describing the direction of travel at every frame of the video
  with a pitch and yaw angle in radians.
- 5 videos are unlabeled. It is your task to generate the labels for them.

Deliverable
-----

Your deliverable is the 5 labels called 5.npy to 9.npy. Zip them up and e-mail it to givemeajob@comma.ai.

Evaluation
-----

We will evaluate your mean squared error. Errors for frames where the car speed is less than 1m/s will be ignored.
Those are also labeled as NaN in the example labels.

Context
------



Twitter
------

<a href="https://twitter.com/comma_ai">Follow us!</a>

