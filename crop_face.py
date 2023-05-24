import cv2
import os
import numpy as np
from tqdm import tqdm

# Load the pre-trained s3fd face detection model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# Set CUDA as the preferred backend and target
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Create output folders for cropped images and debug images
os.makedirs("output/face_cropped", exist_ok=True)
os.makedirs("output/face_debug", exist_ok=True)

# Get a list of input image paths
input_folder = "input"
output_folder = "output/face_cropped"
debug_folder = "output/face_debug"
image_paths = [os.path.join(input_folder, file) for file in os.listdir(input_folder)]

# Initialize progress bar
progress_bar = tqdm(total=len(image_paths), desc="Processing images")

# Margins for the upper body crop (in percentage)
left_margin_percent = 75
top_margin_percent = 75
right_margin_percent = 75
bottom_margin_percent = 75

# Process each image
for image_path in image_paths:
    # Load the image
    image = cv2.imread(image_path)
    
    # Perform face detection
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    # Crop and resize upper body for each detected face
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (startX, startY, endX, endY) = box.astype(int)
            
            # Calculate the width and height of the bounding box
            width = endX - startX
            height = endY - startY
            
            # Calculate the margins for each side of the upper body crop
            left_margin = width * left_margin_percent // 100
            top_margin = height * top_margin_percent // 100
            right_margin = width * right_margin_percent // 100
            bottom_margin = height * bottom_margin_percent // 100
            
            # Calculate the coordinates of the upper body region
            upper_left_x = max(startX - left_margin, 0)
            upper_left_y = max(startY - top_margin, 0)
            lower_right_x = min(endX + right_margin, image.shape[1])
            lower_right_y = min(endY + bottom_margin, image.shape[0])
            
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
            
            # Resize to 1080x1080 pixels
            resized_image = cv2.resize(square_region, (1080, 1080))
            
            # Calculate the thickness of the rectangle based on the image resolution
            resolution_thickness_ratio = image.shape[1] // 1000
            thickness = max(resolution_thickness_ratio, 5)
            
            # Draw rectangle on debug image
            debug_image = image.copy()
            cv2.rectangle(debug_image, (square_upper_left_x, square_upper_left_y), (square_lower_right_x, square_lower_right_y), (0, 255, 0), thickness)

            # Add text label with original image resolution
            resolution_text = f"{image.shape[1]}x{image.shape[0]} face_{i}"

            # Set the background color and text color
            background_color = (0, 0, 0)  # Black color for the background
            text_color = (255, 255, 255)  # White color for the text

            # Calculate the size of the text label
            text_size, _ = cv2.getTextSize(resolution_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

            # Create a rectangular background for the text
            background = np.zeros((text_size[1] + 10, text_size[0] + 10, 3), dtype=np.uint8)
            background[:, :] = background_color

            # Add the text label on top of the background
            text_position = (10, 30 + text_size[1])
            cv2.putText(background, resolution_text, (10, text_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 4)

            # Overlay the background with the text on the debug image
            debug_image[0:text_size[1] + 10, 0:text_size[0] + 10] = background

            # Resize debug image to half the resolution
            debug_image = cv2.resize(debug_image, (image.shape[1] // 2, image.shape[0] // 2))
            
            # Save the resized image
            output_path = os.path.join(output_folder, f"cropped_{i}_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, resized_image)
            
            # Save debug image
            debug_path = os.path.join(debug_folder, f"debug_{i}_{os.path.basename(image_path)}")
            cv2.imwrite(debug_path, debug_image)
    
    # Update progress bar
    progress_bar.update(1)

# Finish progress bar
progress_bar.close()