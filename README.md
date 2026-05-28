# PRO357 GIVING MACHINES VISION USING MACHINE LEARNING

## PURPOSE
This program is intended to identify the contents of a belt that transports shredded foam pieces. We attempt at identifying how much Bitumin, Aluminium or EPS may be found as impurities on the belt alongside Foam.

## DATASET

We have 2 cameras angled left and right of the conveyor belt who's images (384x384, RGB) will be saved in seperate folders called CAMERA_N where N is the index of the camera (here 0 to 1). The images will be extracted from videos using the naming-scheme: `CameraId_VideoId_{CLASSIFICATION}_{CLASSIFICATION}_{CLASSIFICATION}_{CLASSIFICATION}`. The images themselves will be named in the scheme `CameraId_VideoId_FrameId_FOAM_{CLASSIFICATION}_BITUMIN_{CLASSIFICATION}_ALUMINIUM_{CLASSIFICATION}_EPS_{CLASSIFICATION}` where classifications are inherited from the video labels and can be `[HIGH, MEDIUM, LOW, NONE]`. The video's classifications are manually assigned.

The labeled dataset is split into a training batch (80%) to train the model, a validation batch to fine tune the model (10%) and a testing batch (10%) in order to test the model on unseen images. 

Data leaks are expected with minimal footage, but limited with p-hashing to avoid near-duplicate images.

## PROGRAM STRUCTURE
The images will be extracted from videos, classified and distributed to the training, testing and validations subsets using **dataset.py**.
The program is split into 3 scripts, where the **input.py** script represents the user-interface and **cnn.py** is the main working part of the program. The user-interface is loading the dataset whilst the convolutional neural-network handles the machine learning required to identify the state of the belt's contents.

