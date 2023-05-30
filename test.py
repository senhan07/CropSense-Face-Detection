import os
import numpy as np
from tqdm import tqdm
from image_processing import process_image
from user_input import select_option, preview_window, clean_output
import variable

os.makedirs("output/upperbody_cropped", exist_ok=True)
os.makedirs("output/upperbody_debug", exist_ok=True)
os.makedirs("output/face_cropped", exist_ok=True)
os.makedirs("output/face_debug", exist_ok=True)
os.makedirs("output/fullbody_cropped", exist_ok=True)
os.makedirs("output/fullbody_debug", exist_ok=True)
os.makedirs("output/error_images", exist_ok=True)

input_folder = variable.input_folder
output_res = variable.output_res
error_folder = variable.error_folder
preview_output_res = variable.preview_output_res
preview_debug_max_res = variable.preview_debug_max_res
image_paths = [os.path.join(input_folder, file) for file in os.listdir(input_folder)]

if __name__ == '__main__':

    top_margin_value,bottom_margin_value, debug_output,output_folder, croptype = select_option()
    show_preview = preview_window()
    clean_output(output_folder, debug_output, error_folder)
    
    process_image(input_folder,
                  error_folder,
                  output_folder,
                  debug_output,
                  output_res,
                  preview_output_res,
                  preview_debug_max_res,
                  show_preview,
                  croptype, 
                  top_margin_value, 
                  bottom_margin_value)