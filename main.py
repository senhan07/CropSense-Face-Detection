import cv2
import os
import time
import shutil
import imghdr
import numpy as np
from tqdm import tqdm

# Load the pre-trained s3fd face detection model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# # Set CUDA as the preferred backend and target
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Create output folders for cropped images, debug images, and error images
os.makedirs("output/upperbody_cropped", exist_ok=True)
os.makedirs("output/upperbody_debug", exist_ok=True)
os.makedirs("output/face_cropped", exist_ok=True)
os.makedirs("output/face_debug", exist_ok=True)
os.makedirs("output/fullbody_cropped", exist_ok=True)
os.makedirs("output/fullbody_debug", exist_ok=True)
os.makedirs("output/error_images", exist_ok=True)

# Output cropped resolution (pixel)
output_res = 1080
preview_output_res = 256
preview_debug_res = 768

# Minimun face bounding box size (pixel)
min_face_res = 128
min_upperbody_res = 96
min_fullbody_res = 32

# Set confidence level
confidence_level = 0.8

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

# Set initial variable
output_folder = ""
debug_output = ""
top_margin_value = ""
bottom_margin_value = ""
processed_images = 0
boundingbox_class = 0
error_count = 0

# User input for selecting the crop size
while True:
    option = input("Select a crop type:\n1. Upper Body\n2. Face\n3. Full Body\nSelect: ")
    if option not in ["1", "2", "3"]:
        print("Invalid option selected. Please try again.")
        print("")
    else:
        break
# Define margin values based on the selected option
if option == "1":
    top_margin_value = 0.5
    bottom_margin_value = 2.5
    debug_output = debug_upperbody_folder
    output_folder = output_upperbody_folder
    boundingbox_class = 1
elif option == "2":
    top_margin_value = 0.5
    bottom_margin_value = 0.5
    debug_output = debug_face_folder
    output_folder = output_face_folder
    boundingbox_class = 2
elif option == "3":
    top_margin_value = 0.5
    bottom_margin_value = 6
    debug_output = debug_fullbody_folder
    output_folder = output_fullbody_folder
    boundingbox_class = 3

# User input for enabling preview window
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

# User input for deleting output folder
while len(os.listdir(output_folder)) > 0:
    # Check if the output folder is not empty
    file_exist = input("Clean the output folder? [Y]es/[N]o: ")
    if file_exist.lower() == "y":
        print("Cleaning the output folder...")
        
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

print("Starting...")

# Initialize progress bar
progress_bar = tqdm(total=len(image_paths))

# Process each image
for image_path in image_paths:
    filename, extension = os.path.splitext(os.path.basename(image_path))

    # Load the image
    image = cv2.imread(image_path)

    # Check the file format using imghdr
    image_format = imghdr.what(image_path)

    # Verify if the image format is supported
    supported_formats = ["jpg", "jpeg", "png", "webp"]  # Add more formats as needed
    
    if image_format is None or image_format not in supported_formats:
        filename, extension = os.path.splitext(os.path.basename(image_path))
        print(f"\rInvalid image format or unsupported format, skipping {filename}{extension}")
        error_count += 1
    else:
        # Perform face detection
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        def images_error():
            # Copy the original image to the error images folder
            shutil.copy2(image_path, error_folder)

        # Check if the face is error
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Check if the face bellow the confidence level
            if confidence < confidence_level:
                print(f"\rConfidence level too low ({int(confidence * 100)}%), skipping {filename}{extension}")
                images_error()
                error_count += 1
                break

            # Filter out weak detections
            if confidence > confidence_level:

                box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                (startX, startY, endX, endY) = box.astype(int)

                # Calculate the width and height of the bounding box
                width = endX - startX
                height = endY - startY

                # Check if the width or height is too small
                if boundingbox_class == 3:
                    if width < min_fullbody_res or height < min_fullbody_res:
                        print(f"\rFace resolution is too small for fullbody crop, skipping {filename}{extension}")
                        images_error()
                        error_count += 1
                        break
                elif boundingbox_class == 2:
                    if width < min_face_res or height < min_face_res:
                        print(f"\rFace resolution is too small for face crop, skipping {filename}{extension}")
                        images_error()
                        error_count += 1
                        break
                elif boundingbox_class == 1:
                    if width < min_upperbody_res or height < min_upperbody_res:
                        print(f"\rFace resolution is too small for upperbody crop, skipping {filename}{extension}")
                        images_error()
                        error_count += 1
                        break
            break
        # Crop and resize upper body for each detected face
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence > confidence_level:
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
                    background_color = (255, 255, 0)  # Cyan color for the background
                    text_color = (0, 0, 0)  # Black color for the text

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

                    max_size = preview_debug_res
                    # Calculate the new width and height while preserving the aspect ratio
                    width = image.shape[1]
                    height = image.shape[0]

                    if width > height:
                        new_width = min(width, max_size)
                        new_height = int(height * new_width / width)
                    else:
                        new_height = min(height, max_size)
                        new_width = int(width * new_height / height)

                    # Resize the image
                    debug_image = cv2.resize(debug_image, (new_width, new_height))

                    # Save the debug image with rectangle and label
                    filename = os.path.splitext(os.path.basename(image_path))[0]
                    debug_image_path = os.path.join(debug_output, f"{filename}_face_{i}.jpg")
                    cv2.imwrite(debug_image_path, debug_image)

                    # Save the cropped and resized image
                    output_image_path = os.path.join(output_folder, f"{filename}_face_{i}.png")
                    if cv2.imwrite(output_image_path, resized_image):
                        processed_images += 1

                    if show_preview == True:
                        # Define the desired maximum and minimum width and height of the preview window
                        debug_max_window_width = 768
                        debug_max_window_height = 768
                        debug_min_window_width = 400
                        debug_min_window_height = 400
    
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
                        
                        cv2.waitKey(250)  # Wait time     

    # Update progress bar
    progress_bar.set_description(f'Error images: {error_count}, Processing images')
    progress_bar.update(1)

# Finish progress bar
progress_bar.close()

# Close preview windows
cv2.destroyAllWindows()  

# Calculate the total number of input images
total_images = len(image_paths)

processed_images = total_images - error_count

# Print the statistics
print(f"Total images: {total_images}")
print(f"Processed images: {processed_images}")
print(f"Error images: {error_count}")