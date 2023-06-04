import os
import cv2
from image_processing import process_image, draw_rectangle, cropping, preview
from user_input import select_option, preview_window, clean_output
from face_landmark import calculate_face_quality
import variable
from tqdm import tqdm

input_folder = variable.input_folder
output_res = variable.output_res
error_folder = variable.error_folder
preview_output_res = variable.preview_output_res
preview_debug_max_res = variable.preview_debug_max_res
image_paths = [os.path.join(input_folder, file) for file in os.listdir(input_folder)]
processed_images = 0
error_count = 0

if __name__ == '__main__':
    #Ask crop type
    top_margin_value,bottom_margin_value, debug_output,output_folder, croptype = select_option()
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(debug_output, exist_ok=True)
    os.makedirs(error_folder, exist_ok=True)

    #Ask enable preview window
    show_preview = preview_window()
    clean_output(output_folder, debug_output, error_folder)
    
    progress_bar = tqdm(total=len(image_paths), desc="Processing images", dynamic_ncols=True)
    
    #Process face detection
    for image_path in image_paths:
        endX, \
        startX, \
        endY, \
        startY, \
        top_margin_value, \
        bottom_margin_value, \
        image, \
        output_res, \
        output_folder, \
        output_image_path, \
        filename, \
        image_path, \
        debug_output, \
        show_preview, \
        preview_output_res, \
        preview_debug_max_res, \
        is_error, \
        i, \
        confidence, \
        error_msg = process_image(image_path,
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
        
        #Return face and cropping coordinate 
        result_cropping = cropping(endX, \
                            startX, \
                            endY, \
                            startY,
                            top_margin_value, \
                            bottom_margin_value, \
                            image, \
                            output_folder, \
                            filename, \
                            i,)

        square_region = result_cropping[0]
        output_image_path = result_cropping[1]
        image = result_cropping[2]
        startX = result_cropping[3]
        startY = result_cropping[4]
        endX = result_cropping[5]
        endY = result_cropping[6]
        square_upper_left_x = result_cropping[7]
        square_upper_left_y = result_cropping[8]
        square_lower_right_x = result_cropping[9]
        square_lower_right_y = result_cropping[10]
        square_margin_upper_left_x = result_cropping[11]
        square_margin_upper_left_y = result_cropping[12]
        square_margin_size = result_cropping[13]
        i = result_cropping[14]
        width_square = result_cropping[15]

        result_quality = calculate_face_quality(image, startX, startY, endX, endY, is_error)
        image = result_quality[0]
        x = result_quality[1]
        y = result_quality[2]
        quality = result_quality[3]
        shape = result_quality[4]


        debug_image, resized_image = draw_rectangle(square_region, \
                                                    is_error, \
                                                    output_res, \
                                                    output_image_path, \
                                                    image, \
                                                    startX, \
                                                    startY, \
                                                    endX, \
                                                    endY, \
                                                    square_upper_left_x, \
                                                    square_upper_left_y, \
                                                    square_lower_right_x, \
                                                    square_lower_right_y, \
                                                    square_margin_upper_left_x, \
                                                    square_margin_upper_left_y, \
                                                    square_margin_size, \
                                                    i, \
                                                    width_square, \
                                                    confidence, \
                                                    error_msg, \
                                                    image_path, \
                                                    debug_output, \
                                                    x, \
                                                    y, \
                                                    shape, \
                                                    quality)
        

        if show_preview == True:
            preview(debug_image,
                resized_image,
                preview_output_res,
                preview_debug_max_res,
                is_error)
        
        
        
        
        if is_error == True:
            is_error += 1
        processed_images += 1
        progress_bar.update(1)






    cv2.destroyAllWindows()  
    progress_bar.close()
    total_images = len(image_paths)
    total_output_images = len(os.listdir(output_folder))
    print(f"Total images: {total_images}")
    print(f"Processed images: {processed_images}")
    print(f"Total faces detected: {total_output_images}")
    print(f"Error images: {error_count}")