import cv2
import os
import imghdr
import win32com.client
from tqdm import tqdm
import numpy as np
import variable
from user_input import select_option

def images_error(image_path, error_folder):
    shell = win32com.client.Dispatch("WScript.Shell")
    filename_shortcut = os.path.basename(image_path)
    shortcut_path = os.path.join(error_folder, filename_shortcut + ".lnk")
    shortcut = shell.CreateShortcut(shortcut_path)
    shortcut.TargetPath = os.path.abspath(image_path)
    shortcut.Save()



def process_image(input_folder, error_folder, output_res):
    option = select_option()
    if option == "1":
        top_margin_value = 1
        bottom_margin_value = 3
        debug_output = variable.debug_upperbody_folder
        output_folder = variable.output_upperbody_folder
        boundingbox_class = 1
    elif option == "2":
        top_margin_value = 1
        bottom_margin_value = 1
        debug_output = variable.debug_face_folder
        output_folder = variable.output_face_folder
        boundingbox_class = 2
    elif option == "3":
        top_margin_value = 3
        bottom_margin_value = 5
        debug_output = variable.debug_fullbody_folder
        output_folder = variable.output_fullbody_folder
        boundingbox_class = 3

    net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
    image_paths = [os.path.join(input_folder, file) for file in os.listdir(input_folder)]
    progress_bar = tqdm(total=len(image_paths), desc="Processing images")
    for image_path in image_paths:
        is_error = False
        filename, extension = os.path.splitext(os.path.basename(image_path))
        image = cv2.imread(image_path)
        image_format = imghdr.what(image_path)
        supported_formats = ["jpg", "jpeg", "png", "webp"]
        if image_format is None or image_format not in supported_formats:
            print(f"\rInvalid image format or unsupported format, skipping {filename}{extension}")
            variable.error_count += 1
        else:
            if image.shape[0] > 300 or image.shape[1] > 300:
                blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                net.setInput(blob)
                detections = net.forward()

                # Check if the face is error
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]

                    # Check if the face bellow the confidence level
                    if confidence < variable.confidence_level:
                        print(f"Confidence level too low ({int(confidence * 100)}%), skipping face_{i} on {filename}{extension}")
                        images_error(image_path, error_folder)
                        is_error = True
                        variable.error_count += 1
                        break

                    # Filter out weak detections
                    if confidence > variable.confidence_level:

                        box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                        (startX, startY, endX, endY) = box.astype(int)

                        # Calculate the width and height of the bounding box
                        width = endX - startX
                        height = endY - startY

                        # Check if the width or height is too small
                        if boundingbox_class == 3:
                            if width < variable.min_fullbody_res or height < variable.min_fullbody_res:
                                print(f"\rFace resolution is too small for fullbody crop, skipping {filename}{extension}")
                                images_error(image_path, error_folder)
                                is_error = True
                                variable.error_count += 1
                                break
                        elif boundingbox_class == 2:
                            if width < variable.min_face_res or height < variable.min_face_res:
                                print(f"\rFace resolution is too small for face crop, skipping {filename}{extension}")
                                images_error(image_path, error_folder)
                                is_error = True
                                variable.error_count += 1
                                break
                        elif boundingbox_class == 1:
                            if width < variable.min_upperbody_res or height < variable.min_upperbody_res:
                                print(f"\rFace resolution is too small for upperbody crop, skipping {filename}{extension}")
                                images_error(image_path, error_folder)
                                is_error = True
                                variable.error_count += 1
                                break
                    
                    # Calculate the size of the square region to be cropped
                    square_size = min(endX - startX, endY - startY)

                    # Calculate the coordinates for the square region
                    square_upper_left_x = (endX + startX) // 2 - square_size // 2
                    square_upper_left_y = (endY + startY) // 2 - square_size // 2
                    square_lower_right_x = square_upper_left_x + square_size
                    square_lower_right_y = square_upper_left_y + square_size

                    # SECOND BOX MARGIN
                    width_square = square_lower_right_x - square_upper_left_x
                    height_square = square_lower_right_y - square_upper_left_y

                    # Calculate the margin based on the height of the bounding box
                    top_margin_percent = int(height_square * top_margin_value) 
                    bottom_margin_percent = int(height_square * bottom_margin_value)
                    left_margin_percent = bottom_margin_percent
                    right_margin_percent = bottom_margin_percent
                    
                    # Calculate the coordinates of the upper body region
                    upper_left_x = max(square_upper_left_x - left_margin_percent, 0)
                    upper_left_y = max(square_upper_left_y - top_margin_percent, 0)
                    lower_right_x = min(square_lower_right_x + right_margin_percent, image.shape[1])
                    lower_right_y = min(square_lower_right_y + bottom_margin_percent, image.shape[0])
                    
                    square_margin_size = min(lower_right_x - upper_left_x, lower_right_y - upper_left_y)

                    # Calculate the coordinates for the square region
                    square_margin_upper_left_x = (lower_right_x + upper_left_x) // 2 - square_margin_size // 2
                    square_margin_upper_left_y = (lower_right_y + upper_left_y) // 2 - square_margin_size // 2
                    square_margin_lower_right_x = square_margin_upper_left_x + square_margin_size
                    square_margin_lower_right_y = square_margin_upper_left_y + square_margin_size

                    # Cropped image
                    square_region = image[square_margin_upper_left_y:square_margin_lower_right_y, square_margin_upper_left_x:square_margin_lower_right_x]

                    # Check if the square region is valid (not empty)
                    if square_region.size == 0:
                        continue
                    
                    # Resize output image
                    resized_image = cv2.resize(square_region, (output_res, output_res))

                    # Save the cropped and resized image
                    output_image_path = os.path.join(output_folder, f"{filename}_face_{i}.png")
                    if is_error == False:
                        cv2.imwrite(output_image_path, resized_image)

            else:
                print(f"\rThe resolution is too low for face detection, skipping {filename}{extension}")
                images_error(image_path, error_folder)
                variable.error_count += 1


def show_preview(debug_image, resized_image, preview_output_res, preview_debug_max_res, preview_debug_min_res):
    # Define the desired maximum and minimum width and height of the preview window
    debug_max_window_width = preview_debug_max_res
    debug_max_window_height = preview_debug_max_res
    debug_min_window_width = preview_debug_min_res
    debug_min_window_height = preview_debug_min_res

    output_preview_res = preview_output_res

    # Resize the debug image to fit within the maximum window dimensions while maintaining the aspect ratio
    window_width = debug_image.shape[1]
    window_height = debug_image.shape[0]
    window_aspect_ratio = window_width / float(window_height)

    # Debug preview images
    if window_width > debug_max_window_width or window_height > debug_max_window_height:
        # Check if the width or height exceeds the maximum limits
        width_scale_factor = debug_max_window_width / window_width
        height_scale_factor = debug_max_window_height / window_height
        scale_factor = min(width_scale_factor, height_scale_factor)
    else:
        # Check if the width or height is below the minimum limits
        width_scale_factor = debug_min_window_width / window_width
        height_scale_factor = debug_min_window_height / window_height
        scale_factor = max(width_scale_factor, height_scale_factor)

    new_width = int(window_width * scale_factor)
    new_height = int(window_height * scale_factor)

    debug_preview_image = cv2.resize(debug_image, (new_width, new_height))
    output_preview_image = cv2.resize(resized_image, (output_preview_res, output_preview_res))

    # Show a preview window of the debug image and set it to stay on top
    cv2.namedWindow("Debug Image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Debug Image", debug_preview_image)
    cv2.setWindowProperty("Debug Image", cv2.WND_PROP_TOPMOST, 1)
    cv2.setWindowProperty("Debug Image", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

    # Show a preview window of the output image and set it to stay on top
    cv2.namedWindow("Output Image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Output Image", output_preview_image)
    cv2.setWindowProperty("Output Image", cv2.WND_PROP_TOPMOST, 1)
    cv2.setWindowProperty("Output Image", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    
    cv2.waitKey(0)  # Wait time   
