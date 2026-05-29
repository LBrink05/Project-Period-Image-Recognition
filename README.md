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


## Videos classes
1. pirpur
2. eps
3. pirpural1
4. pirpural2
5. pirpurmixeverything
6. pirpurbitumen
7. pirpureps
8. pirpurmoreal1
9. pirpurmoreal2
10. pirpurmorebitumen
We filmed these 10 batches, with 6 videos per batch, getting around 5-10 seconds per video.

for the different batches: 
for pirpur control is HIGH NONE NONE NONE, 
for eps control is NONE NONE NONE HIGH, 
for pirpur al1 is MEDIUM NONE LOW NONE 
for pirpur al2 is MEDIUM, NONE, LOW, NONE,
for pirpurmixeverything varies so please judge yourself, 
for pirpurbitumin is MEDIUM LOW NONE NONE 
for pirpureps is MEDIUM NONE NONE MEDIUM,
for pirpurmoreal1 and for pirpurmoreal2 judge yourself, 
for pirpurmorebitumin Is MEDIUM MEDIUM NONE NONE
