# PRO357 GIVING MACHINES VISION USING MACHINE LEARNING

## PURPOSE
This program is intended to identify the contents of a belt that transports shredded foam pieces. We attempt at identifying how much Bitumin, Aluminium or EPS may be found as impurities on the belt alongside Foam.

## DATASET

We have 2 cameras angled left and right of the conveyor belt who's images (384x384, RGB) will be saved in seperate folders called CAMERA_N where N is the index of the camera (here 0 to 1). The images will be extracted from videos using the naming-scheme: `CameraId_VideoName_VideoCount_{CLASSIFICATION}_{CLASSIFICATION}_{CLASSIFICATION}_{CLASSIFICATION}`. The images themselves will be named in the scheme `CameraId_VideoName_VideoCount_FrameId_FOAM_{CLASSIFICATION}_BITUMEN_{CLASSIFICATION}_ALUMINIUM_{CLASSIFICATION}_EPS_{CLASSIFICATION}` where classifications are inherited from the video labels and can be `[HIGH, MEDIUM, LOW, NONE]`. The video's classifications are manually assigned.

The labeled dataset is split into a training batch (80%) to train the model, a validation batch to fine tune the model (10%) and a testing batch (10%) in order to test the model on unseen images. 

Data leaks are expected with minimal footage, but limited with p-hashing to avoid near-duplicate images.

## VIDEO BATCHES
We filmed these 10 batches, with 6 videos per batch, getting around 5-10 seconds per video.

1. pirpur - control sample of pirpur
2. eps - control sample of eps
3. pirpural1 - mix of pirpur and aluminium (lighter foam)
4. pirpural2 - mix of pirpur and aluminium (darker, more connected foam)
5. pirpurmixeverything - mix of everything
6. pirpurbitumen - mix of pirpur and bitumen
7. pirpureps - mix of pirpur and eps
8. pirpurmoreal1 - mix of pirpur and more aluminium (lighter foam)
9. pirpurmoreal2 - mix of pirpur and more aluminium (darker, more connected foam)
10. pirpurmorebitumen - mix of pirpur and more bitumen

For the manual classifications we abided by these guidelines:
Classification order is FOAM, BITUMEN, ALUMINIUM, EPS.

pirpur: HIGH NONE NONE NONE, 
eps: NONE NONE NONE HIGH, 
pirpural1: MEDIUM NONE LOW NONE 
pirpural2: MEDIUM NONE LOW NONE,
pirpurmixeverything judge yourself, 
pirpurbitumen: MEDIUM LOW NONE NONE 
pirpureps: MEDIUM NONE NONE MEDIUM,
pirpurmoreal1 and pirpurmoreal2 judge yourself, 
pirpurmorebitumen: MEDIUM MEDIUM NONE NONE

## PROGRAM STRUCTURE
The program is split into 4 scripts. Images will be extracted from videos, classified and distributed to the training, testing and validations subsets using **dataset.py**.
The **input.py** script represents the user-interface and **cnn.py** is the main working part of the program. The user-interface is loading the dataset whilst the convolutional neural-network handles the machine learning required to identify the state of the belt's contents. **output.py** deals with the graphical illustrations of the results computed by the neural network.

## NEURAL NETWORK STRUCTURE

The network is a multiheaded convolutional network trained using mini-batches of 50 images and backpropagation at a learning rate of 0.0005 for 50 epochs. The loss function used to train models is categorical cross-entropy as it proportionally punishes the model to wrong its answers are. For every category classification, the loss function results are weighted by frequency of training examples in the training set. In addition, when pairing softmax with cross-entropy (canonical pairing), the required derivative of the loss function for backpropagation simplifies to `pred - true`.
It consists of these layers: Conv -> ReLU -> MaxPool -> Conv -> ReLu -> MaxPool -> Dense -> GroupedSoftMax

What the Layers do:
- Convolutional Layer extracts features using valid convolution 
- ReLU (Rectified Linear Unit) activation layer transforms features (clamps negative to 0)
- MaxPool reduces the amount of values within an image of features by compressing it by taking the max value within a 2x2 window which slides across the image.
- Dense layer computes scores for every option i.e. Foam: Medium etc. (called logits)
- GroupedSoftMax applies softmax function to every material independently (4 independent softmax classifiers)

## TODO
Take videos of samples IMPORTANT:

  - foam = LOW
  - bitumen = HIGH
  - aluminium = HIGH
  - eps = LOW
  - bitumen = MEDIUM 
  - eps = MEDIUM 