import os
from image_processing import process_image
from user_input import select_option, preview_window, clean_output
import variable

input_folder = variable.input_folder
output_res = variable.output_res
error_folder = variable.error_folder
preview_output_res = variable.preview_output_res
preview_debug_max_res = variable.preview_debug_max_res

if __name__ == '__main__':

    top_margin_value,bottom_margin_value, debug_output,output_folder, croptype = select_option()
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(debug_output, exist_ok=True)
    os.makedirs(error_folder, exist_ok=True)

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