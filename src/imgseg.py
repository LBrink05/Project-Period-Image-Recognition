import cnn
import random
import cv2

# We are tying to do deep feature clustering with Gaussian Mixture Model
#used in order to fine-tune image segmentation used for detection purposes
def demo(network, path):
    print("demo")

    #Load Random Image
    data, labels = cnn.load_dataset(path)

    i = random.randrange(len(data))
    image, true = data[i], labels[i]

    inputi = image[None]

    #Extract Features with network
    out = network[0].forward(inputi) #conv1
    out = network[1].forward(out) #ReLU
    out = network[2].forward(out) #MaxPool
    out = network[3].forward(out) #conv2
    out = network[4].forward(out) #ReLu
    out = network[5].forward(out) #MaxPool

    #Showing original sample image

    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original Image', 512, 512)
    cv2.imshow('Original Image', (image.transpose(1,2,0) * 255).astype('uint8'))
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


