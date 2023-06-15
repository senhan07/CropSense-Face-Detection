import cv2
import os
import imghdr
from torch import align_tensors
import win32com.client
from tqdm import tqdm
import numpy as np
import variable
import dlib
import math

def images_error(image_path, error_folder):
    shell = win32com.client.Dispatch("WScript.Shell")
    filename_shortcut = os.path.basename(image_path)
    shortcut_path = os.path.join(error_folder, filename_shortcut + ".lnk")
    shortcut = shell.CreateShortcut(shortcut_path)
    shortcut.TargetPath = os.path.abspath(image_path)
    shortcut.Save()

def detect_landmarks(image, bbox):
    # Load the shape predictor model
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path) # type: ignore

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a dlib rectangle object from the bounding box coordinates
    dlib_rect = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3]) # type: ignore

    # Perform face landmark detection
    landmarks = predictor(gray, dlib_rect)

    # Convert the landmarks to a numpy array
    landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

    return landmarks


def rotate_bbox(bbox, center, angle):
    # Convert the angle to radians
    angle_rad = math.radians(angle)

    # Calculate the sine and cosine of the angle
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    # Calculate the coordinates of the rotated bounding box
    rotated_bbox = []
    for x, y in [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]:
        # Shift the coordinate system to the rotation center
        x_shifted = x - center[0]
        y_shifted = y - center[1]

        # Apply the rotation transformation
        rotated_x = center[0] + cos_theta * x_shifted - sin_theta * y_shifted
        rotated_y = center[1] + sin_theta * x_shifted + cos_theta * y_shifted

        # Append the rotated coordinates to the list
        rotated_bbox.append(int(rotated_x))
        rotated_bbox.append(int(rotated_y))

    return rotated_bbox



def process_image(image_path,
                  error_folder,
                  output_folder,
                  debug_output,
                  output_res,
                  preview_output_res,
                  preview_debug_max_res,
                  show_preview,
                  croptype, 
                  top_margin_value, 
                  bottom_margin_value,):
    error_count = 0
    output_image_path = ""
    error_msg = ""

    net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
    is_error = False
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

                if confidence < variable.confidence_level:
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
                
                if confidence > variable.confidence_level:
                    if croptype == 3:
                        if width < variable.min_fullbody_res or height < variable.min_fullbody_res:
                            print(f"\rFace resolution is too small for fullbody crop, skipping face_{i} on {filename}{extension}")
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
                    elif croptype == 2:
                        if width < variable.min_face_res or height < variable.min_face_res:
                            print(f"\rFace resolution is too small for face crop, skipping face_{i} on {filename}{extension}")
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
                    elif croptype == 1:
                        if width < variable.min_upperbody_res or height < variable.min_upperbody_res:
                            print(f"\rFace resolution is too small for upperbody crop, skipping face_{i} on {filename}{extension}")
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
                break
            for i in range(detections.shape[2]):
                if is_error == True:
                    break
                confidence = detections[0, 0, i, 2]

                # Filter out weak detections
                if confidence > variable.confidence_level:
                    box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                    (startX, startY, endX, endY) = box.astype(int)
                
                    width = endX - startX
                    height = endY - startY

                    is_error = draw_rectangle(endX,
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
                                        error_msg
                                        )
                else:
                    break
        else:
            print(f"\rThe resolution is too low for face detection, skipping {filename}{extension}")
            images_error(image_path, error_folder)
            error_count += 1
    return error_count

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
                   error_msg
                   ):
    
    # Calculate the size of the square region to be cropped
    square_size = min(endX - startX, endY - startY)

    # Calculate the coordinates for the square region
    square_center_x = (endX + startX) // 2
    square_center_y = (endY + startY) // 2



    # Calculate the angle of rotation based on the eye landmarks
    landmarks = detect_landmarks(image, (startX, startY, endX, endY))
    eye_left_x, eye_left_y = landmarks[36]
    eye_right_x, eye_right_y = landmarks[45]
    dx = eye_right_x - eye_left_x
    dy = eye_right_y - eye_left_y
    angle = math.degrees(math.atan2(dy, dx))

    # Calculate the rotation matrix
    rotation_center = (square_center_x, square_center_y)
    rotation_matrix = cv2.getRotationMatrix2D(rotation_center, angle, 1.0)

    # Rotate the four corners of the bounding box
    corners = np.array([[startX, startY], [endX, startY], [endX, endY], [startX, endY]], dtype=np.float32)
    rotated_corners = cv2.transform(corners.reshape(-1, 1, 2), rotation_matrix).reshape(-1, 2)

    # Find the new bounding box coordinates
    rotated_startX = int(np.min(rotated_corners[:, 0]))
    rotated_startY = int(np.min(rotated_corners[:, 1]))
    rotated_endX = int(np.max(rotated_corners[:, 0]))
    rotated_endY = int(np.max(rotated_corners[:, 1]))


    square_upper_left_x = rotated_startX
    square_upper_left_y = rotated_startY
    square_lower_right_x = rotated_endX
    square_lower_right_y = rotated_endY





    width_square = square_lower_right_x - square_upper_left_x
    height_square = square_lower_right_y - square_upper_left_y
    
    # Calculate the margin based on the height of the bounding box
    top_margin_percent = int(square_size * top_margin_value) 
    bottom_margin_percent = int(square_size * bottom_margin_value)
    left_margin_percent = int(square_size * bottom_margin_value)
    right_margin_percent = int(square_size * bottom_margin_value)

    # Calculate the coordinates of the upper body region
    upper_left_x = max(square_upper_left_x - left_margin_percent, 0)
    upper_left_y = max(square_upper_left_y - top_margin_percent, 0)  # Align to the top of the face

    # Calculate the lower right coordinates based on the maximum image dimensions
    lower_right_x = min(square_lower_right_x + right_margin_percent, image.shape[1])
    lower_right_y = min(square_lower_right_y + bottom_margin_percent, image.shape[0])

    # Adjust the upper_left_x and lower_right_x based on the width of the cropped region
    width_square_margin = lower_right_x - upper_left_x
    if width_square_margin > image.shape[1]:
        shift_amount = width_square_margin - image.shape[1]
        upper_left_x = max(upper_left_x - shift_amount, 0)
        lower_right_x = image.shape[1]

    # Adjust the upper_left_y based on the height of the cropped region
    height_square_margin = lower_right_y - upper_left_y
    if height_square_margin > image.shape[0]:
        shift_amount = height_square_margin - image.shape[0]
        upper_left_y = max(upper_left_y - shift_amount, 0)
        lower_right_y = image.shape[0]

    # Calculate the new square size based on the adjusted coordinates
    square_margin_size = min(lower_right_x - upper_left_x, lower_right_y - upper_left_y)

    # Calculate the coordinates for the square region
    square_margin_center_x = (lower_right_x + upper_left_x) // 2
    square_margin_upper_left_x = square_margin_center_x - square_margin_size // 2
    square_margin_upper_left_y = upper_left_y  # Align to the top of the face
    square_margin_lower_right_x = square_margin_upper_left_x + square_margin_size
    square_margin_lower_right_y = square_margin_upper_left_y + square_margin_size

    # Calculate the center point of the second box
    second_box_center_x = (startX + endX) // 2

    # Calculate the shift amount for aligning the square_region horizontally
    shift_amount = second_box_center_x - ((square_margin_upper_left_x + square_margin_lower_right_x) // 2)

    # Adjust the square_margin_upper_left_x and square_margin_lower_right_x based on the shift amount
    square_margin_upper_left_x += shift_amount
    square_margin_lower_right_x += shift_amount

    # Crop the image to the square region with margin
    square_margin_upper_left_x = max(square_margin_upper_left_x, 0)
    square_margin_lower_right_x = min(square_margin_lower_right_x, image.shape[1])

    # Adjust square_margin_size based on the final coordinates
    square_margin_size = square_margin_lower_right_x - square_margin_upper_left_x

    square_region = image[square_margin_upper_left_y:square_margin_upper_left_y + square_margin_size, square_margin_upper_left_x:square_margin_upper_left_x + square_margin_size]

    # Save the cropped and resized image
    output_image_path = os.path.join(output_folder, f"{filename}_face_{i}.png") # type: ignore
    resized_image = ""

    if square_region.size == 0:
        is_error = True
        error_msg = ("NO FACE DETECTED")
    else:
        if is_error == False:
            resized_image = cv2.resize(square_region, (output_res, output_res))
            cv2.imwrite(output_image_path, resized_image)
    
    # Calculate the thickness of the rectangle based on the image resolution
    resolution_thickness_ratio = image.shape[1] // 128
    thickness = max(resolution_thickness_ratio, 5)

    debug_image = image.copy()

    cv2.rectangle(debug_image, (startX, startY), (endX, endY), (0, 0, 255), thickness) # type: ignore #face rectangle
    cv2.rectangle(debug_image, (square_upper_left_x, square_upper_left_y), (square_lower_right_x, square_lower_right_y), (0, 255, 0), thickness) # type: ignore #crop rectagle
    cv2.rectangle(debug_image, (square_margin_upper_left_x, square_margin_upper_left_y), # type: ignore
              (square_margin_upper_left_x + square_margin_size, square_margin_upper_left_y + square_margin_size),
              (255,165,0), thickness) 
        # Draw the face landmarks on the image
    for (x, y) in landmarks:
        cv2.circle(debug_image, (x, y), 10, (0, 255, 0), -1)   

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
    return is_error

def preview(debug_image,
            resized_image,
            preview_output_res,
            preview_debug_max_res,
            is_error):
    
    # Resize the debug image to fit within the maximum window dimensions while maintaining the aspect ratio
    window_width = debug_image.shape[1]
    window_height = debug_image.shape[0]

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

    cv2.namedWindow("Debug Image", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Debug Image", debug_preview_image)
    cv2.setWindowProperty("Debug Image", cv2.WND_PROP_TOPMOST, 1)
    cv2.setWindowProperty("Debug Image", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

    if is_error == False:
        output_preview_image = cv2.resize(resized_image, (preview_output_res, preview_output_res))
        cv2.namedWindow("Output Image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Output Image", output_preview_image)
        cv2.setWindowProperty("Output Image", cv2.WND_PROP_TOPMOST, 1)
        cv2.setWindowProperty("Output Image", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        
    cv2.waitKey(0)  # Wait time   