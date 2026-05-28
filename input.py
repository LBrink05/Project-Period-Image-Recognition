## PRO357 GIVING MACHINES VISION USING MACHINE LEARNING

# import cnn.py (place holder until have cnn.py)

from pathlib import Path
import numpy as np
import os
import cv2
import shutil
from colorama import Fore, Back, Style
import sys

TRAINING_PATH = Path("Data/final/training")
TESTING_PATH = Path("Data/final/testing")
VALIDATING_PATH = Path("Data/final/validating")

terminal_char_len = shutil.get_terminal_size(fallback=(80, 24)).columns

def shutting_down(statement="", style=""):
    print(style + statement + Style.RESET_ALL)
    print("\n" + "#"*terminal_char_len)
    print("Shutting down interface program. \n")
    sys.exit()

def load_dataset(folder: Path):
    images, labels = [], []
    for file in sorted(folder.glob("*.jpg")):
        img = cv2.imread(str(file))
        img = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        img = img.transpose(2, 0, 1)  # Change to (C, H, W) format
        images.append(img)

        parts = file.stem.split("_")
        label = {
            "foam":      parts[parts.index("FOAM") + 1],
            "bitumin":   parts[parts.index("BITUMIN") + 1],
            "aluminium": parts[parts.index("ALUMINIUM") + 1],
            "eps":       parts[parts.index("EPS") + 1],
        }
        labels.append(label)

    data = np.array(images)
    return data, labels
            
def train():
    print("\n[Training]")
    print("Loading training dataset...")
    train_data, train_labels = load_dataset(TRAINING_PATH)
    print(f"Loaded {len(train_labels)} training images, shape: {train_data.shape}.")
    print(Fore.GREEN + "Training started...\n" + Style.RESET_ALL)
     # TODO:call cnn.train(train_data, train_labels)

def test():
    print("\n[Testing]")
    print("Loading testing dataset...")
    test_data, test_labels = load_dataset(TESTING_PATH)
    print(f"Loaded {len(test_labels)} testing images, shape: {test_data.shape}.")
    print(Fore.GREEN + "Testing started...\n" + Style.RESET_ALL)
     # TODO:call cnn.test(test_data, test_labels)

def validate():
    print("\n[Validating]")
    print("Loading validating dataset...")
    val_data, val_labels = load_dataset(VALIDATING_PATH)
    print(f"Loaded {len(val_labels)} validating images, shape: {val_data.shape}.")
    print(Fore.GREEN + "Validating started...\n" + Style.RESET_ALL)
    # TODO:call cnn.val(val_data, val_labels)

def menu():
    print("-"*terminal_char_len)
    print("PRO357 - Giving Machines Vision Using Machine Learning")
    print("-"*terminal_char_len)
    print("(1) Train model")
    print("(2) Test model")
    print("(3) Validate model")
    print("(4) Exit")
    print("-"*terminal_char_len)

def main():
    while True:
        menu()
        choice = input("Please select an option: ").strip()
        if choice == "1":
            train()
        elif choice == "2":
            test()
        elif choice == "3":
            validate()
        elif choice == "4":
            shutting_down("Exiting program :)", Fore.YELLOW)
            break
        else:
            print(Fore.RED + "\nErroneous user-input received." + Style.RESET_ALL)
            print(Fore.YELLOW + "Please enter 1, 2 or 3." + Style.RESET_ALL)


# program executed directly only runs main
if __name__ == "__main__":
    main()
 