from pathlib import Path
import pathlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import time
from collections import deque
import sys
import shutil
from PIL import Image
import imagehash

# constants & nearly unchanging variables
RAW_DATA_PATH = Path("Data/unlabeled")
DATA_PATH_STR = "Data/unlabeled"
ALLOWED_FLAGS = ["a","s","d","f"] #Much, Some, Little, None
terminal_char_len = shutil.get_terminal_size(fallback=(80, 24)).columns

def shutting_down():
    print("\n" + "#"*terminal_char_len)
    print("Shutting down dataset preparation program. \n")
    sys.exit()
 
def extract_images():
    for file in RAW_DATA_PATH.rglob('*.mp4'):
        if file.is_file():
            frameid = 0
            videoid = file.stem
            camid = file.parent.parent.name
            video = cv2.VideoCapture(file)
            recent_hashes = deque(maxlen=5)
            filepath = DATA_PATH_STR + f"/{camid}/{videoid}/"
            pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)

            #Remove any prior generated images in cami/videoi directory
            for f in Path(filepath).glob('*'):
                if f.is_file():
                    f.unlink()

            while True:
                ret, frame = video.read()
                if not ret:
                    break

                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                hash_o = imagehash.phash(pil_frame)

                if any(hash_o - h < 5 for h in recent_hashes):
                    frameid += 1
                    continue

                name = f"{filepath}{camid}_{videoid}_{frameid}.jpg"
                cv2.imwrite(name, frame)
                recent_hashes.append(hash_o)
                print('Extracting: ' + name)
                frameid += 1
            

            video.release()
            cv2.destroyAllWindows()
            print(f"Successfully extraced frames from {camid} {videoid}")
            shutting_down()

def classify_images():
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
                    match foam_flag:
                        case 'a':
                            foam_flag = "HIGH"
                        case 's':
                            foam_flag = "MEDIUM"
                        case 'd':
                            foam_flag = "LOW"
                        case 'f': 
                            foam_flag = "None"
                else:
                    print("Erroneous user-input received for foam.")
                    break

                impure_flag = input("Rate the amount of impurities: [Much: a, Some: s, Little: d, None: f]")

                if impure_flag in ALLOWED_FLAGS:
                    match impure_flag:
                        case 'a':
                            impure_flag = "HIGH"
                        case 's':
                            impure_flag = "MEDIUM"
                        case 'd':
                            impure_flag = "LOW"
                        case 'f': 
                            impure_flag = "None"
                else:
                    print("Erroneous user-input received for impurities.")
                    break

                break
            
            camid, videoid, frameid =  file.stem.split('_')
            changed_path = "Data/labeled/"
            pathlib.Path(changed_path).mkdir(parents=True, exist_ok=True)
            changed_filepath = changed_path +  f"/{camid}_{videoid}_{frameid}" + "_FOAM_" + foam_flag + "_IMPURITIES_" + impure_flag + ".jpg"
            pathlib.Path(changed_filepath).touch()
    
    shutting_down()


#extracting image frames from videos
while True:
    query = input("Do you wish to extract images from videos? (y/n): ")
    print("\n")

    if query == 'yes' or query == 'y':
        extract_images()
        
    elif query == 'no' or query == 'n':
        break


while True: 
    query = input("Do you wish to classify images? (y/n): ")
    print("\n")
    if query == 'yes' or query == 'y':
        classify_images()
        
    elif query == 'no' or query == 'n':
        break

while True:
    query = input("Do you wish to ")
shutting_down()

