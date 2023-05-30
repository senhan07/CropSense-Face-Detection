import cv2
import os
import time
import shutil
import imghdr
import win32com.client
import numpy as np
from tqdm import tqdm
from image_processing import process_image
from user_input import select_option
import variable

os.makedirs("output/upperbody_cropped", exist_ok=True)
os.makedirs("output/upperbody_debug", exist_ok=True)
os.makedirs("output/face_cropped", exist_ok=True)
os.makedirs("output/face_debug", exist_ok=True)
os.makedirs("output/fullbody_cropped", exist_ok=True)
os.makedirs("output/fullbody_debug", exist_ok=True)
os.makedirs("output/error_images", exist_ok=True)

input_folder = "input"
image_paths = [os.path.join(input_folder, file) for file in os.listdir(input_folder)]

output_res = variable.output_res
error_folder = variable.error_folder

if __name__ == '__main__':
    
    process_image(input_folder, error_folder, output_res)