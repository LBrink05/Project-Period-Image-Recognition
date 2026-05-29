## PRO357 GIVING MACHINES VISION USING MACHINE LEARNING

from pathlib import Path
import numpy as np
import os
import cv2
import shutil
from colorama import Fore, Back, Style
import sys

import cnn
import output

TRAINING_PATH = Path("Data/final/training")
TESTING_PATH = Path("Data/final/testing")
VALIDATING_PATH = Path("Data/final/validating")

terminal_char_len = shutil.get_terminal_size(fallback=(80, 24)).columns

def shutting_down(statement="", style=""):
    print(style + statement + Style.RESET_ALL)
    print("\n" + "#"*terminal_char_len)
    print("Shutting down interface program. \n")
    sys.exit()

#loading data batch wise to not run out of memory, hence Path used for functions
          
def train():
    print("\n[Training]")
    print(Fore.GREEN + "Training started...\n" + Style.RESET_ALL)
    cnn.train(TRAINING_PATH)
    print(Fore.GREEN + "Training successfully completed.\n" + Style.RESET_ALL)

def test():
    print("\n[Testing]")
    print(Fore.GREEN + "Testing started...\n" + Style.RESET_ALL)
    cnn.test(TESTING_PATH)
    print(Fore.GREEN + "Testing successfully completed.\n" + Style.RESET_ALL)

def validate():
    print("\n[Validating]")
    print(Fore.GREEN + "Validating started...\n" + Style.RESET_ALL)
    cnn.val(VALIDATING_PATH)
    print(Fore.GREEN + "Validating successfully completed.\n" + Style.RESET_ALL)

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
 