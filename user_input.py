import os
import time
import variable
from tqdm import tqdm

def select_option():
    output_upperbody_folder = variable.output_upperbody_folder
    debug_upperbody_folder = variable.debug_upperbody_folder
    output_face_folder = variable.output_face_folder
    debug_face_folder = variable.debug_face_folder
    output_fullbody_folder = variable.output_fullbody_folder
    debug_fullbody_folder = variable.debug_fullbody_folder

    while True:
        option = input("Select a crop type:\n1. Upper Body\n2. Face\n3. Full Body\nSelect: ")
        if option not in ["1", "2", "3"]:
            print("Invalid option selected. Please try again.")
            print("")
        else:
            break
    if option == "1":
        top_margin_value = 1
        bottom_margin_value = 3
        debug_output = debug_upperbody_folder
        output_folder = output_upperbody_folder
        croptype = 1
    elif option == "2":
        top_margin_value = 1
        bottom_margin_value = 1
        debug_output = debug_face_folder
        output_folder = output_face_folder
        croptype = 2
    elif option == "3":
        top_margin_value = 3
        bottom_margin_value = 5
        debug_output = debug_fullbody_folder
        output_folder = output_fullbody_folder
        croptype = 3
    return [top_margin_value, #type: ignore
            bottom_margin_value, #type: ignore
            debug_output, #type: ignore
            output_folder, #type: ignore
            croptype] #type: ignore

def preview_window():
    show_preview = False
    while True:
        show_preview = input("Show preview window? [Y]es/[N]o: ")
        if show_preview.lower() == "y":
            show_preview = True
            break
        elif show_preview.lower() == "n":
            show_preview = False
            break
        else:
            print("Invalid option selected. Please enter 'Y' or 'N'.")
            print("")
    return show_preview

def clean_output(output_folder, debug_output, error_folder):
    while len(os.listdir(output_folder)) > 0:
        file_exist = input("Output folders are not empty,  clean it? [Y]es/[N]o: ")
        if file_exist.lower() == "y":
            # Add a delay of 5 seconds before proceeding with the deletion
            print("Deleting files in 5... [PRESS CTRL+C TO CANCEL]")
            time.sleep(1)
            print("Deleting files in 4... [PRESS CTRL+C TO CANCEL]")
            time.sleep(1)
            print("Deleting files in 3... [PRESS CTRL+C TO CANCEL]")
            time.sleep(1)
            print("Deleting files in 2... [PRESS CTRL+C TO CANCEL]")
            time.sleep(1)
            print("Deleting files in 1... [PRESS CTRL+C TO CANCEL]")
            time.sleep(1)

            # Get the total number of files
            total_files = sum([len(files) for _, _, files in os.walk(output_folder)])

            # Deleting files in the output folder and subdirectories with a progress bar
            with tqdm(total=total_files, unit='file') as pbar:
                for root, dirs, files in os.walk(output_folder):
                    for file_name in files:
                        file_path = os.path.join(root, file_name)
                        os.remove(file_path)
                        pbar.update(1)

            with tqdm(total=total_files, unit='file') as pbar:
                for root, dirs, files in os.walk(debug_output):
                    for file_name in files:
                        file_path = os.path.join(root, file_name)
                        os.remove(file_path)
                        pbar.update(1)

            with tqdm(total=total_files, unit='file') as pbar:
                for root, dirs, files in os.walk(error_folder):
                    for file_name in files:
                        file_path = os.path.join(root, file_name)
                        os.remove(file_path)
                        pbar.update(1)

            print("Done cleaning the output folder.")
            print("")
        elif file_exist.lower() == "n":
            print("")
            break  # Exit the while loop
        else:
            print("Invalid option selected. Please enter 'Y' or 'N'.")
            print("")