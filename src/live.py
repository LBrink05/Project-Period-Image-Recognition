import cv2
from matplotlib import pyplot as plt
import time
from PIL import Image
import numpy as np

import cnn
from cnn import MATERIALS, center_crop_size
import dataset

def take_photo():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cv2.imwrite('webcamphoto.jpg', frame)
    cap.release()

#Note: To use with phone, use IP Webcam App and use http://192.168.2.5:8080/video for address

def live_video(network, address):

    try:
        cap = cv2.VideoCapture(0)
    except Exception:
        print("Incorrect camera url given")
        return

    while cap.isOpened():

        time.sleep(0.2) #decide how often to sample images for analysis #to not overheat computer
        ret, frame = cap.read(address)

        img = process_live_images(frame)
        pred = cnn.test_image(network, img)
        
        classifications = "| " + "".join(f"{mat.capitalize()} = {pred[mat].capitalize()} | " for mat in MATERIALS)
        text_color = (166, 30, 214)
        cv2.putText(frame, classifications, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA)

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_live_images(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
    new_width, new_height = img.size

    if new_width < new_height:
        new_height = new_width
    else: 
        new_width = new_height

    #resizing images to set resolution (1:1) of 128x128
    img = dataset.center_crop(img, new_width, new_height)
    img = img.resize((center_crop_size, center_crop_size))

    img = (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
    return img
    