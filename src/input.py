## PRO357 GIVING MACHINES VISION USING MACHINE LEARNING

from pathlib import Path
import numpy as np
import os
import cv2
import shutil
from colorama import Fore, Back, Style
import sys
import pickle
import cnn

import output
import live

ALL_PATH = Path("Data/final/all")
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
          
def start_train(network):
    print("\n[Training]")
    print(Fore.GREEN + "Training started...\n" + Style.RESET_ALL)
    query = input("Enter amount of epochs (empty=25): ")
    if query == "":
        cnn.train(TRAINING_PATH, VALIDATING_PATH, network)
    else:
        cnn.train(TRAINING_PATH, VALIDATING_PATH, network, epochs=query)
    print(Fore.GREEN + "Training successfully completed.\n" + Style.RESET_ALL)

def start_test(network):
    print("\n[Testing]")
    print(Fore.GREEN + "Testing started...\n" + Style.RESET_ALL)
    cnn.test(TESTING_PATH, network)
    print(Fore.GREEN + "Testing successfully completed.\n" + Style.RESET_ALL)

def start_validate(network):
    print("\n[Validating]")
    print(Fore.GREEN + "Validating started...\n" + Style.RESET_ALL)
    cnn.val(VALIDATING_PATH, network)
    print(Fore.GREEN + "Validating successfully completed.\n" + Style.RESET_ALL)

def start_loading_model(network):
    query = input("Which model would you like to load? (Empty = model.pkl): ")
    if query != "":
        cnn.load_network(network, query)
    else:
        cnn.load_network(network)

    print(Fore.GREEN + "Loaded model successfully." + Style.RESET_ALL)

def start_save_model(network):
    query = input("What would you like to name the model? (Empty = model.pkl): ")
    if query != "":
        cnn.save_network(network, query)
        print(Fore.GREEN + f"Successfully saved model {query}." + Style.RESET_ALL)
    else:
        cnn.save_network(network)
        print(Fore.GREEN + f"Successfully saved model.pkl" + Style.RESET_ALL)

def start_live_video(network):
    query = input("Do you wish to start a live video feed (y/n)?: ")
    if query == 'y' or query == 'yes':
        print(Fore.GREEN + "Starting live video feed..." + Style.RESET_ALL)
        query = input("Do you wish to use your phone? If so, enter url: ")
        if query != "":
            live.live_video(network, query)
        else:
            live.live_video(network)
        print(Fore.YELLOW + "Stopping live video feed." + Style.RESET_ALL)
    

def menu():
    print("-"*terminal_char_len)
    print("PRO357 - Giving Machines Vision Using Machine Learning")
    print("-"*terminal_char_len)
    print("(1) Train model")
    print("(2) Test model")
    print("(3) Save model")
    print("(4) Load model")
    print("(5) Live Video")
    print("(6) Exit")
    print("-"*terminal_char_len)

def main():

    print("Initiating network...")
    network = cnn.build_network()
    print(Fore.YELLOW + "Network successfully initiated." + Style.RESET_ALL)

    while True:
        menu()
        choice = input("Please select an option: ").strip()
        if choice == "1":
            start_train(network)
        elif choice == "2":
            start_test(network)
        elif choice == "3":
            start_save_model(network)
        elif choice == "4":
            start_loading_model(network)
        elif choice == "5":
            start_live_video(network)
        elif choice == "6":
            shutting_down("Exiting program :)", Fore.YELLOW)
            break
        else:
            print(Fore.RED + "\nErroneous user-input received." + Style.RESET_ALL)
            print(Fore.YELLOW + "Please enter 1-6" + Style.RESET_ALL)


# program executed directly only runs main
if __name__ == "__main__":
    main()
 