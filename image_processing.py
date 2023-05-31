import cv2
import os
import imghdr
import win32com.client
from tqdm import tqdm
import numpy as np
import variable

def images_error(image_path, error_folder):
    shell = win32com.client.Dispatch("WScript.Shell")
    filename_shortcut = os.path.basename(image_path)
    shortcut_path = os.path.join(error_folder, filename_shortcut + ".lnk")
    shortcut = shell.CreateShortcut(shortcut_path)
    shortcut.TargetPath = os.path.abspath(image_path)
    shortcut.Save()



def process_image(input_folder,
                  error_folder,
                  output_folder,
                  debug_output,
                  output_res,
                  preview_output_res,
                  preview_debug_max_res,
                  show_preview,
                  croptype, 
                  top_margin_value, 
                  bottom_margin_value):
    error_count = 0
    endX = 0
    endY = 0
    startX = 0
    startY = 0
    i = 0
    confidence = 0
    image = ""
    image_path = ""
    is_error = False
    output_image_path = ""
    filename = ""
    error_msg = ""

    net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
    image_paths = [os.path.join(input_folder, file) for file in os.listdir(input_folder)]
    progress_bar = tqdm(total=len(image_paths), desc="Processing images")
    for image_path in image_paths:
        filename, extension = os.path.splitext(os.path.basename(image_path))
        image = cv2.imread(image_path)
        image_format = imghdr.what(image_path)
        supported_formats = ["jpg", "jpeg", "png", "webp"]
        if image_format is None or image_format not in supported_formats:
            print(f"\rInvalid image format or unsupported format, skipping {filename}{extension}")
            error_count += 1
        else:
            if image.shape[0] > 300 or image.shape[1] > 300:
                blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                net.setInput(blob)
                detections = net.forward()

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                    (startX, startY, endX, endY) = box.astype(int)

                    width = endX - startX
                    height = endY - startY
                    if confidence > variable.confidence_level:

                        if croptype == 3:
                            if width < variable.min_fullbody_res or height < variable.min_fullbody_res:
                                print(f"\rFace resolution is too small for fullbody crop, skipping {filename}{extension}")
                                error_msg = "FACE RESOLUTION IS TOO SMALL"
                                images_error(image_path, error_folder)
                                is_error = True
                                error_count += 1
                                draw_rectangle(endX,
                                                startX,
                                                endY,
                                                startY,
                                                top_margin_value,
                                                bottom_margin_value,
                                                image, 
                                                output_res,
                                                output_folder,
                                                output_image_path,
                                                filename,
                                                image_path,
                                                debug_output,
                                                show_preview,
                                                preview_output_res,
                                                preview_debug_max_res,
                                                is_error,
                                                i,
                                                confidence,
                                                error_msg)
                        elif croptype == 2:
                            if width < variable.min_face_res or height < variable.min_face_res:
                                print(f"\rFace resolution is too small for face crop, skipping {filename}{extension}")
                                error_msg = "FACE RESOLUTION IS TOO SMALL"
                                images_error(image_path, error_folder)
                                is_error = True
                                error_count += 1
                                draw_rectangle(endX,
                                                startX,
                                                endY,
                                                startY,
                                                top_margin_value,
                                                bottom_margin_value,
                                                image, 
                                                output_res,
                                                output_folder,
                                                output_image_path,
                                                filename,
                                                image_path,
                                                debug_output,
                                                show_preview,
                                                preview_output_res,
                                                preview_debug_max_res,
                                                is_error,
                                                i,
                                                confidence,
                                                error_msg)
                        elif croptype == 1:
                            if width < variable.min_upperbody_res or height < variable.min_upperbody_res:
                                print(f"\rFace resolution is too small for upperbody crop, skipping {filename}{extension}")
                                error_msg = "FACE RESOLUTION IS TOO SMALL"
                                images_error(image_path, error_folder)
                                is_error = True
                                error_count += 1
                                draw_rectangle(endX,
                                                startX,
                                                endY,
                                                startY,
                                                top_margin_value,
                                                bottom_margin_value,
                                                image, 
                                                output_res,
                                                output_folder,
                                                output_image_path,
                                                filename,
                                                image_path,
                                                debug_output,
                                                show_preview,
                                                preview_output_res,
                                                preview_debug_max_res,
                                                is_error,
                                                i,
                                                confidence,
                                                error_msg)
                        break
                    else:
                        print(f"\rConfidence level too low ({int(confidence * 100)}%), skipping face_{i} on {filename}{extension}")
                        error_msg = "CONFIDENCE LEVEL TOO LOW"
                        images_error(image_path, error_folder)
                        is_error = True
                        error_count += 1
                        draw_rectangle(endX,
                                        startX,
                                        endY,
                                        startY,
                                        top_margin_value,
                                        bottom_margin_value,
                                        image, 
                                        output_res,
                                        output_folder,
                                        output_image_path,
                                        filename,
                                        image_path,
                                        debug_output,
                                        show_preview,
                                        preview_output_res,
                                        preview_debug_max_res,
                                        is_error,
                                        i,
                                        confidence,
                                        error_msg)
                        break
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]

                    # Filter out weak detections
                    if confidence > variable.confidence_level:
                        box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                        (startX, startY, endX, endY) = box.astype(int)
                    
                        width = endX - startX
                        height = endY - startY
                        draw_rectangle(endX,
                                            startX,
                                            endY,
                                            startY,
                                            top_margin_value,
                                            bottom_margin_value,
                                            image, 
                                            output_res,
                                            output_folder,
                                            output_image_path,
                                            filename,
                                            image_path,
                                            debug_output,
                                            show_preview,
                                            preview_output_res,
                                            preview_debug_max_res,
                                            is_error,
                                            i,
                                            confidence,
                                            error_msg)
            else:
                print(f"\rThe resolution is too low for face detection, skipping {filename}{extension}")
                images_error(image_path, error_folder)
                error_count += 1
        progress_bar.update(1)
    cv2.destroyAllWindows()  
    progress_bar.close()

    total_images = len(image_paths)
    total_output_images = len(os.listdir(output_folder))
    processed_images = total_images - error_count
    print(f"Total images: {total_images}")
    print(f"Processed images: {processed_images}")
    print(f"Total faces: {total_output_images}")
    print(f"Error images: {error_count}")
    return [endX,
            startX,
            endY,
            startY,
            top_margin_value,
            bottom_margin_value,
            image, 
            output_res,
            output_folder,
            output_image_path,
            filename,
            image_path,
            debug_output,
            show_preview,
            preview_output_res,
            preview_debug_max_res,
            is_error,
            i,
            confidence,
            error_msg]


def draw_rectangle(endX,
                   startX,
                   endY,
                   startY,
                   top_margin_value,
                   bottom_margin_value,
                   image, 
                   output_res,
                   output_folder,
                   output_image_path,
                   filename,
                   image_path,
                   debug_output,
                   show_preview,
                   preview_output_res,
                   preview_debug_max_res,
                   is_error,
                   i,
                   confidence,
                   error_msg):
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
    # if square_region.size == 0:
        # Resize output image

    # Save the cropped and resized image
    output_image_path = os.path.join(output_folder, f"{filename}_face_{i}.png") # type: ignore
    resized_image = ""
    if is_error == False:
        resized_image = cv2.resize(square_region, (output_res, output_res))
        cv2.imwrite(output_image_path, resized_image)

    # Calculate the thickness of the rectangle based on the image resolution
    resolution_thickness_ratio = image.shape[1] // 128
    thickness = max(resolution_thickness_ratio, 5)

    # Draw rectangle on debug image
    debug_image = image.copy()

    cv2.rectangle(debug_image, (startX, startY), (endX, endY), (0, 0, 255), thickness) #face rectangle
    cv2.rectangle(debug_image, (square_upper_left_x, square_upper_left_y), (square_lower_right_x, square_lower_right_y), (0, 255, 0), thickness) #crop rectagle
    cv2.rectangle(debug_image, (square_margin_upper_left_x, square_margin_upper_left_y), (square_margin_lower_right_x, square_margin_lower_right_y), (255, 165, 0), thickness)

    font_scale = min(image.shape[1], image.shape[0]) / 1000
    font_thickness = max(1, int(min(image.shape[1], image.shape[0]) / 500))


    # FIRST TEXT LABEL
    resolution_text = f"{image.shape[1]}x{image.shape[0]} face_{i}_{width_square}px ({int(confidence * 100)}%)"
    background_color = (255, 255, 0)
    text_color = (0, 0, 0)
    text_size, _ = cv2.getTextSize(resolution_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

    background_width = text_size[0] + 10
    background_height = text_size[1] + 10
    background = np.zeros((background_height, background_width, 3), dtype=np.uint8)
    background[:, :] = background_color


    # Resize the background if its width is greater than the available width in debug_image
    if background_width > debug_image.shape[1]:
        ratio = debug_image.shape[1] / background_width
        background_width = debug_image.shape[1]
        background_height = int(background_height * ratio)
        background = cv2.resize(background, (background_width, background_height))

    cv2.putText(background, resolution_text, (10, text_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
    debug_image[0:background_height, 0:background_width] = background


    # SECOND TEXT LABEL
    if is_error == True:
        second_background_color = (0, 0, 255) 
        second_text_color = (0, 0, 0)
        second_text = f"ERROR {error_msg}"
    else:
        second_background_color = (0, 255, 0) 
        second_text_color = (0, 0, 0) 
        second_text = "OK PASSED"

    second_font_scale = font_scale
    second_font_thickness = font_thickness
    second_text_size, _ = cv2.getTextSize(second_text, cv2.FONT_HERSHEY_SIMPLEX, second_font_scale, second_font_thickness)

    second_background_width = second_text_size[0] + 10
    second_background_height = second_text_size[1] + 10

    second_background = np.zeros((second_background_height, second_background_width, 3), dtype=np.uint8)
    second_background[:, :] = second_background_color

    if second_background_width > debug_image.shape[1]:
        second_ratio = debug_image.shape[1] / second_background_width
        second_background_width = debug_image.shape[1]
        second_background_height = int(second_background_height * second_ratio)
        second_background = cv2.resize(second_background, (second_background_width, second_background_height))

    cv2.putText(second_background, second_text, (10, second_text_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, second_font_scale, second_text_color, second_font_thickness)

    debug_image_height, debug_image_width, _ = debug_image.shape
    combined_height = background_height + second_background_height
    if combined_height > debug_image_height:
        debug_image = cv2.resize(debug_image, (debug_image_width, combined_height))
    debug_image[background_height:combined_height, 0:second_background_width] = second_background



    filename = os.path.splitext(os.path.basename(image_path))[0]
    debug_image_path = os.path.join(debug_output, f"{filename}_face_{i}.jpg")
    cv2.imwrite(debug_image_path, debug_image)

    if show_preview == True:
        preview(debug_image, resized_image, preview_output_res, preview_debug_max_res, is_error)

def preview(debug_image,
            resized_image,
            preview_output_res,
            preview_debug_max_res,
            is_error):
    
    # Resize the debug image to fit within the maximum window dimensions while maintaining the aspect ratio
    window_width = debug_image.shape[1]
    window_height = debug_image.shape[0]
    window_aspect_ratio = window_width / float(window_height)

    # Debug preview images
    if window_width > preview_debug_max_res or window_height > preview_debug_max_res:
        # Check if the width or height exceeds the maximum limits
        width_scale_factor = preview_debug_max_res / window_width
        height_scale_factor = preview_debug_max_res / window_height
        scale_factor = min(width_scale_factor, height_scale_factor)
    else:
        # Check if the width or height is below the minimum limits
        width_scale_factor = preview_output_res / window_width
        height_scale_factor = preview_output_res / window_height
        scale_factor = max(width_scale_factor, height_scale_factor)

    new_width = int(window_width * scale_factor)
    new_height = int(window_height * scale_factor)

    debug_preview_image = cv2.resize(debug_image, (new_width, new_height))

    # Show a preview window of the debug image and set it to stay on top
    cv2.namedWindow("Debug Image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Debug Image", debug_preview_image)
    cv2.setWindowProperty("Debug Image", cv2.WND_PROP_TOPMOST, 1)
    cv2.setWindowProperty("Debug Image", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

    if is_error == False:
        output_preview_image = cv2.resize(resized_image, (preview_output_res, preview_output_res))
        # Show a preview window of the output image and set it to stay on top
        cv2.namedWindow("Output Image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Output Image", output_preview_image)
        cv2.setWindowProperty("Output Image", cv2.WND_PROP_TOPMOST, 1)
        cv2.setWindowProperty("Output Image", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        
    cv2.waitKey(0)  # Wait time   