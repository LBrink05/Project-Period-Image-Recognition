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
from colorama import Fore, Back, Style
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

def shutting_down(statement="", style=""):
    print(style + statement + Style.RESET_ALL)
    print("\n" + "#"*terminal_char_len)
    print("Shutting down dataset preparation program. \n")
    sys.exit()
    
def extract_labels(camid, videoid):
    # Expecting: cam0_videoname_foam_bitumen_aluminium_eps.mp4
    matches = list(RAW_DATA_PATH.rglob(f'{camid}_{videoid}_*.mp4'))
    if not matches:
        raise FileNotFoundError(f"No video found for {camid}_{videoid}")
    if len(matches) > 1:
        raise ValueError(f"Multiple videos match {camid}_{videoid}: {matches}")

    parts = matches[0].stem.split('_')
    if len(parts) < 6:
        raise ValueError(f"Unexpected filename format: {matches[0].name}")

    foam, bitumen, aluminium, eps = parts[2], parts[3], parts[4], parts[5]
    return foam, bitumen, aluminium, eps

def extract_images():
    for file in RAW_DATA_PATH.rglob('*.mp4'):
        if file.is_file():
            frameid = 0
            videoid = file.stem.split('_')[1]
            camid = file.stem.split('_')[0]
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

    shutting_down("Successfully extraced frames!", Fore.GREEN)

def classify_images_vid():
    input(Fore.YELLOW + "Be aware to manually delete files before retrying classification." + Style.RESET_ALL)

    for file in RAW_DATA_PATH.rglob('*.jpg'):
        if file.is_file():
            print(f"File found: {file}")

            img = mpimg.imread(file)

            camid, videoid, frameid =  file.stem.split('_')
            
            foam_flag, bitumin_flag, aluminium_flag, eps_flag = extract_labels(camid, videoid)

            changed_path = "Data/labeled"
            pathlib.Path(changed_path).mkdir(parents=True, exist_ok=True)
            changed_filepath = changed_path +  f"/{camid}_{videoid}_{frameid}" + "_FOAM_" + foam_flag + "_BITUMIN_" + bitumin_flag + "_AL_" + aluminium_flag + "_EPS_" + eps_flag + ".jpg"
            shutil.copy(file, changed_filepath)
    
    shutting_down("Successfully classified Images", Fore.GREEN)

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

    
    shutting_down("Successfully processed Images!", Fore.GREEN)



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

    classify_images_vid()

            
#processing images
while True:
    query = input("Do you wish to process the classified images? (y/n)")
    print("\n")

    if query == 'yes' or query == 'y':
        process_images()

    elif query == 'no' or query == 'n':
        break


shutting_down("No task was selected.", Fore.RED)

