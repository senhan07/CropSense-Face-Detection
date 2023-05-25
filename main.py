import cv2
import os
import shutil
import numpy as np
from tqdm import tqdm

# Load the pre-trained s3fd face detection model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# Set CUDA as the preferred backend and target
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Create output folders for cropped images, debug images, and error images
os.makedirs("output/upperbody_cropped", exist_ok=True)
os.makedirs("output/upperbody_debug", exist_ok=True)
os.makedirs("output/face_cropped", exist_ok=True)
os.makedirs("output/face_debug", exist_ok=True)
os.makedirs("output/fullbody_cropped", exist_ok=True)
os.makedirs("output/fullbody_debug", exist_ok=True)
os.makedirs("output/error_images", exist_ok=True)

#Output cropped resolution (pixel)
output_res = 1080

# Get a list of input image paths
input_folder = "input"
output_upperbody_folder = "output/upperbody_cropped"
debug_upperbody_folder = "output/upperbody_debug"
output_face_folder = "output/face_cropped"
debug_face_folder = "output/face_debug"
output_fullbody_folder = "output/fullbody_cropped"
debug_fullbody_folder = "output/fullbody_debug"
error_folder = "output/error_images"
image_paths = [os.path.join(input_folder, file) for file in os.listdir(input_folder)]

# User input for selecting the option
option = int(input("Select an option:\n1. Upper Body\n2. Face\n3. Full Body\n"))

# Define margin values based on the selected option
if option == 1:
    top_margin_value = 0.5
    bottom_margin_value = 2.5
    debug_output = debug_upperbody_folder
    output_folder = output_upperbody_folder
elif option == 2:
    top_margin_value = 0.5
    bottom_margin_value = 0.5
    debug_output = debug_face_folder
    output_folder = output_face_folder
elif option == 3:
    top_margin_value = 0.5
    bottom_margin_value = 1
    debug_output = debug_fullbody_folder
    output_folder = output_fullbody_folder
else:
    print("Invalid option selected. Exiting.")
    exit()

# Initialize progress bar
progress_bar = tqdm(total=len(image_paths), desc="Processing images")

# Process each image
for image_path in image_paths:
    # Load the image
    image = cv2.imread(image_path)
    
    # Perform face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    # Flag to check if the face is error
    is_error = False
    
    # Check if the face is error
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections
        if confidence > 0.6:

            box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (startX, startY, endX, endY) = box.astype(int)
            
            # Calculate the width and height of the bounding box
            width = endX - startX
            height = endY - startY
            
            # Check if the width or height is too small
            if width < 64 or height < 64:
                is_error = True
                break
    
    # Skip the image if the face is error
    if is_error:
        # Copy the original image to the error images folder
        shutil.copy2(image_path, error_folder)
        
        # Update progress bar
        progress_bar.update(1)
        continue
    
    # Crop and resize upper body for each detected face
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (startX, startY, endX, endY) = box.astype(int)
            
            # Calculate the width and height of the bounding box
            width = endX - startX
            height = endY - startY
            
            # Calculate the margin based on the height of the bounding box
            top_margin_percent = int(height * top_margin_value) 
            bottom_margin_percent = int(height * bottom_margin_value)
            left_margin_percent = bottom_margin_percent
            right_margin_percent = bottom_margin_percent

            # Calculate the coordinates of the upper body region
            upper_left_x = max(startX - left_margin_percent, 0)
            upper_left_y = max(startY - top_margin_percent, 0)
            lower_right_x = min(endX + right_margin_percent, image.shape[1])
            lower_right_y = min(endY + bottom_margin_percent, image.shape[0])
            
            # Calculate the size of the square region to be cropped
            size = min(lower_right_x - upper_left_x, lower_right_y - upper_left_y)
            
            # Calculate the coordinates for the square region
            square_upper_left_x = (lower_right_x + upper_left_x) // 2 - size // 2
            square_upper_left_y = (lower_right_y + upper_left_y) // 2 - size // 2
            square_lower_right_x = square_upper_left_x + size
            square_lower_right_y = square_upper_left_y + size
            
            # Crop the upper body region as a square
            square_region = image[square_upper_left_y:square_lower_right_y, square_upper_left_x:square_lower_right_x]

            # Check if the square region is valid (not empty)
            if square_region.size == 0:
                continue
            
            # Resize output image
            resized_image = cv2.resize(square_region, (output_res, output_res))
            
            # Calculate the thickness of the rectangle based on the image resolution
            resolution_thickness_ratio = image.shape[1] // 128
            thickness = max(resolution_thickness_ratio, 5)
            
            if endX - startX > 0 and endY - startY > 0:
                # Draw rectangle on debug image
                debug_image = image.copy()
                cv2.rectangle(debug_image, (startX, startY), (endX, endY), (0, 0, 255), thickness) #face rectangle
                cv2.rectangle(debug_image, (square_upper_left_x, square_upper_left_y), (square_lower_right_x, square_lower_right_y), (0, 255, 0), thickness) #crop rectagle
                
                # Add text label with original image resolution and confidence level
                resolution_text = f"{image.shape[1]}x{image.shape[0]} face_{i} ({int(confidence * 100)}%)"
    
                # Set the background color and text color
                background_color = (0, 0, 0)  # Black color for the background
                text_color = (255, 255, 255)  # White color for the text
    
                # Calculate the font scale based on the image resolution
                font_scale = min(image.shape[1], image.shape[0]) / 1000

                # Calculate the font thickness based on the image resolution
                font_thickness = max(1, int(min(image.shape[1], image.shape[0]) / 500))

                # Calculate the size of the text label
                text_size, _ = cv2.getTextSize(resolution_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

                # Calculate the desired width and height for the background
                background_width = text_size[0] + 10
                background_height = text_size[1] + 10

                # Create a rectangular background for the text
                background = np.zeros((background_height, background_width, 3), dtype=np.uint8)
                background[:, :] = background_color

                # Resize the background if its width is greater than the available width in debug_image
                if background_width > debug_image.shape[1]:
                    ratio = debug_image.shape[1] / background_width
                    background_width = debug_image.shape[1]
                    background_height = int(background_height * ratio)
                    background = cv2.resize(background, (background_width, background_height))

                # Add the text label on top of the background
                text_position = (10, 30 + text_size[1])
                cv2.putText(background, resolution_text, (10, text_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

                # Overlay the background with the text on the debug image
                debug_image[0:background_height, 0:background_width] = background

                # Resize debug image to half the resolution
                debug_image = cv2.resize(debug_image, (image.shape[1] // 2, image.shape[0] // 2))
                
                # Save the debug image with rectangle and label
                filename = os.path.splitext(os.path.basename(image_path))[0]
                debug_image_path = os.path.join(debug_output, f"{filename}_face_{i}.jpg")
                cv2.imwrite(debug_image_path, debug_image)
                
                # Save the cropped and resized image
                output_image_path = os.path.join(output_folder, f"{filename}_face_{i}.png")
                cv2.imwrite(output_image_path, resized_image)
            
    # Update progress bar
    progress_bar.update(1)

# Finish progress bar
progress_bar.close()

# Calculate the total number of input images
total_images = len(image_paths)

# Calculate the number of successfully processed images
processed_images = total_images - len(os.listdir(error_folder))

# Calculate the number of skipped images
skipped_images = len(os.listdir(error_folder))

# Print the statistics
print(f"Total images: {total_images}")
print(f"Processed images: {processed_images}")
print(f"Skipped images: {skipped_images}")