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
import random

# constants & nearly unchanging variables
RAW_DATA_PATH = Path("Data/unlabeled")
LABELED_PATH = Path("Data/labeled")
FINAL_PATH = Path("Data/final")

ALLOWED_FLAGS = ["a","s","d","f"] #Much, Some, Little, None
terminal_char_len = shutil.get_terminal_size(fallback=(80, 24)).columns
center_crop_size = 384

def shutting_down(statement=""):
    print(statement)
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
            filepath = str(RAW_DATA_PATH) + f"/{camid}/{videoid}/"
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

                #usually 5, (higher value -> more strict), 0 is no filter
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
            shutting_down(f"Successfully extraced frames from {camid} {videoid}!")

def classify_images():
    print("Be aware to manually delete files before retrying classification.")
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
                            foam_flag = "NONE"
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
                            impure_flag = "NONE"
                else:
                    print("Erroneous user-input received for impurities.")
                    break

                break

            plt.close('all')
            camid, videoid, frameid =  file.stem.split('_')
            changed_path = "Data/labeled"
            pathlib.Path(changed_path).mkdir(parents=True, exist_ok=True)
            changed_filepath = changed_path +  f"/{camid}_{videoid}_{frameid}" + "_FOAM_" + foam_flag + "_IMPURITIES_" + impure_flag + ".jpg"
            shutil.copy(file, changed_filepath)
    
    shutting_down("Successfully classified Images!")

def classify_images_vid():
    print("Be aware to manually delete files before retrying classification.")

    foam_flag = "MISSING"
    impure_flag = "MISSING"

    while True:
        foam_flag_temp = input("Rate the amount of foam: [Much: a, Some: s, Little: d, None: f]")
        
        if foam_flag_temp in ALLOWED_FLAGS:
            match foam_flag_temp:
                case 'a':
                    foam_flag = "HIGH"
                case 's':
                    foam_flag = "MEDIUM"
                case 'd':
                    foam_flag = "LOW"
                case 'f': 
                    foam_flag = "NONE"
        else:
            print(f"Erroneous user-input received for foam: {foam_flag}")
            continue

        impure_flag_temp = input("Rate the amount of impurities: [Much: a, Some: s, Little: d, None: f]")

        if impure_flag_temp in ALLOWED_FLAGS:
            match impure_flag_temp:
                case 'a':
                    impure_flag = "HIGH"
                case 's':
                    impure_flag = "MEDIUM"
                case 'd':
                    impure_flag = "LOW"
                case 'f': 
                    impure_flag = "NONE"
        else:
            print(f"Erroneous user-input received for impurities: {impure_flag}")
            continue
        break

    for file in RAW_DATA_PATH.rglob('*.jpg'):
        if file.is_file():
            print(f"File found: {file}")

            img = mpimg.imread(file)

            camid, videoid, frameid =  file.stem.split('_')
            changed_path = "Data/labeled"
            pathlib.Path(changed_path).mkdir(parents=True, exist_ok=True)
            changed_filepath = changed_path +  f"/{camid}_{videoid}_{frameid}" + "_FOAM_" + foam_flag + "_IMPURITIES_" + impure_flag + ".jpg"
            shutil.copy(file, changed_filepath)
    
    shutting_down("Successfully classified Images")

def center_crop(img, new_width, new_height):
    width, height = img.size
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    img =  img.crop((left, top, right, bottom))
    return img

def process_images():
    folders = ["/all/",  "/validating/", "/testing/", "/training/",]
    imgpath = str(FINAL_PATH) + folders[0]

    for file in LABELED_PATH.glob('*.jpg'):
        if file.is_file():

            #resizing images to set resolution (1:1) of 384×384 
            img = Image.open(file).convert("RGB")
            new_width, new_height = img.size

            if new_width < new_height:
                new_height = new_width
            else: 
                new_width = new_height

            img = center_crop(img, new_width, new_height)
            img = img.resize((384, 384))

            #generalize training set only (rotate, flip, mild color jitter, )
            #DO LATER

            img.save(imgpath + file.name)
    

    #randomly selecting from all and putting it into the other folders

    for file in Path(imgpath).glob('*.jpg'):
        value = random.random()
        if value < 0.1:
            #testing
            shutil.copy(file, str(FINAL_PATH) + folders[2])
        else:
            #training
            shutil.copy(file, str(FINAL_PATH) + folders[3])
    
    for file in Path(str(FINAL_PATH) + folders[3]).glob('*.jpg'):
        value = random.random()
        if value < 0.1:
            #validation
            shutil.copy(file, str(FINAL_PATH) + folders[1])
            
    shutting_down("Successfully processed Images!")



#extracting image frames from videos
while True:
    query = input("Do you wish to extract images from videos? (y/n): ")
    print("\n")

    if query == 'yes' or query == 'y':
        extract_images()
        
    elif query == 'no' or query == 'n':
        break

# Querying if image by image or whole video

#classifying images manually 
while True: 
    query = input("Do you wish to classify images? (y/n): ")
    print("\n")

    if query == 'no' or query == 'n':
        break

    while True:
        subquery = input("Do you wish to classify images by video or per-image? (v/i)")
        if subquery == 'v':
            classify_images_vid()
        elif subquery == 'i':
            classify_images()
    
            
#processing images
while True:
    query = input("Do you wish to process the classified images? (y/n)")
    print("\n")

    if query == 'yes' or query == 'y':
        process_images()

    elif query == 'no' or query == 'n':
        break

shutting_down("No task was selected.")

