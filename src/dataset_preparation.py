from pathlib import Path
import pathlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os

import time

# raw_data path
RAW_DATA_PATH = Path("Data/unlabeled")
DATA_PATH_STR = "Data/unlabeled"

ALLOWED_FLAGS = ["a","s","d","f"] #Much, Some, Little, None

#extracting image frames from videos
while:
    query = input("Do you wish to extract images from videos? (y/n)")
    if query == yes or query == y:
        for file in RAW_DATA_PATH.rglob('*.mp4'):
            if file.is_file():
                frameid = 0
                videoid = file.name[:-4]
                camid = file.parent.parent.name

                video = cv2.VideoCapture(file)
                while(True):
            
                    ret,frame = video.read()

                    if ret:
                        filepath = DATA_PATH_STR + f"/{camid}/" + f"{videoid}/"
                        pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
                        
                        name = filepath + camid + "_" + videoid + "_" + str(frameid) + '.jpg'
                        print ('Creating: ' + name)
                        
                        cv2.imwrite(name, frame)

                        frameid += 1
                    else:
                        break

                
                video.release()
                cv2.destroyAllWindows()
                break
    else if query == no or query == n:
        break


for file in RAW_DATA_PATH.rglob('*.jpg'):
    if file.is_file():
        print(f"File found: {file}")

        img = mpimg.imread(file)
        plt.imshow(img)
        plt.axis('off') 

        while True:

            plt.show(block=False)
            plt.pause(0.1)
            foam_flag = input("Rate the amount of foam: [Much: a, Some: s, Little: d, None: f]")

            if foam_flag in ALLOWED_FLAGS:
                plt.close('all')
                break
            else:
                print("Erroneous user-input received for foam.")

            impure_flag = input("Rate the amount of impurities: [Much: a, Some: s, Little: d, None: f]")

            if impure_flag in ALLOWED_FLAGS:
                plt.close('all')
                break
            else:
                print("Erroneous user-input received for impurities.")
        
        changed_path = file[:-4] + "FOAM_" + foam_flag + "IMPURITIES_" + impure_flag
        pathlib.Path(changed_path).touch()