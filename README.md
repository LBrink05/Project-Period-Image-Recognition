# PRO357 GIVING MACHINES VISION USING MACHINE LEARNING

## DATASET

We have 2 cameras angled left and right of the conveyor belt who's images (384x384, RGB) will be saved in seperate folders called CAMERA_N where N is the index of the camera (here 0 to 1). The images will be extracted from videos using the naming-scheme: `CameraId_VideoId_{CLASSIFICATION}_{CLASSIFICATION}_{CLASSIFICATION}_{CLASSIFICATION}`. The images themselves will be named in the scheme `CameraId_VideoId_FrameId_FOAM_{CLASSIFICATION}_BITUMIN_{CLASSIFICATION}_ALUMINIUM_{CLASSIFICATION}_EPS_{CLASSIFICATION}` where classifications are inherited from the video labels and can be `[HIGH, MEDIUM, LOW, NONE]`. The video's classifications are manually assigned.

The labeled dataset is split into a training batch (80%) to train the model, a validation batch to fine tune the model (10%) and a testing batch (10%) in order to test the model on unseen images. 

Data leaks are expected with minimal footage, but limited with p-hashing to avoid near-duplicate images.

## PROGRAM STRUCTURE
The images will be extracted from videos usi
The program is split into 2 scripts, where the input script handles the queries from the user to the cnn and loading the dataset whilst the cnn script handles the machine learning algorithm part required to identify the state of the belt.

