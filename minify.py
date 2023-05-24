import cv2
import os
import shutil
import numpy as np
from tqdm import tqdm
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
os.makedirs("output/upperbody_cropped", exist_ok=True)
os.makedirs("output/upperbody_debug", exist_ok=True)
os.makedirs("output/face_cropped", exist_ok=True)
os.makedirs("output/face_debug", exist_ok=True)
os.makedirs("output/obstructed_images", exist_ok=True)
input_folder = "input"
output_upperbody_folder = "output/upperbody_cropped"
debug_upperbody_folder = "output/upperbody_debug"
output_face_folder = "output/face_cropped"
debug_face_folder = "output/face_debug"
obstructed_folder = "output/obstructed_images"
image_paths = [os.path.join(input_folder, file) for file in os.listdir(input_folder)]
option = int(input("Select an option:\n1. Upper Body\n2. Face\n"))
if option == 1:
    left_margin_percent = 500
    top_margin_percent = 65
    right_margin_percent = 500
    bottom_margin_percent = 215
elif option == 2:
    left_margin_percent = 75
    top_margin_percent = 75
    right_margin_percent = 75
    bottom_margin_percent = 75
else:
    print("Invalid option selected. Exiting.")
    exit()
progress_bar = tqdm(total=len(image_paths), desc="Processing images")
for image_path in image_paths:
    image = cv2.imread(image_path)
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    is_obstructed = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (startX, startY, endX, endY) = box.astype(int)
            width = endX - startX
            height = endY - startY
            if width < 64 or height < 64:
                is_obstructed = True
                break
    if is_obstructed:
        shutil.copy2(image_path, obstructed_folder)
        progress_bar.update(1)
        continue
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            (startX, startY, endX, endY) = box.astype(int)
            width = endX - startX
            height = endY - startY
            left_margin = width * left_margin_percent // 100
            top_margin = height * top_margin_percent // 100
            right_margin = width * right_margin_percent // 100
            bottom_margin = height * bottom_margin_percent // 100
            upper_left_x = max(startX - left_margin, 0)
            upper_left_y = max(startY - top_margin, 0)
            lower_right_x = min(endX + right_margin, image.shape[1])
            lower_right_y = min(endY + bottom_margin, image.shape[0])
            size = min(lower_right_x - upper_left_x, lower_right_y - upper_left_y)
            square_upper_left_x = (lower_right_x + upper_left_x) // 2 - size // 2
            square_upper_left_y = (lower_right_y + upper_left_y) // 2 - size // 2
            square_lower_right_x = square_upper_left_x + size
            square_lower_right_y = square_upper_left_y + size
            square_region = image[square_upper_left_y:square_lower_right_y, square_upper_left_x:square_lower_right_x]
            if square_region.size == 0:
                continue
            resized_image = cv2.resize(square_region, (1080, 1080))
            resolution_thickness_ratio = image.shape[1] // 1000
            thickness = max(resolution_thickness_ratio, 5)
            debug_image = image.copy()
            cv2.rectangle(debug_image, (square_upper_left_x, square_upper_left_y), (square_lower_right_x, square_lower_right_y), (0, 255, 0), thickness)
            resolution_text = f"{image.shape[1]}x{image.shape[0]} face_{i} ({int(confidence * 100)}%)"
            background_color = (0, 0, 0)  
            text_color = (255, 255, 255)  
            text_size, _ = cv2.getTextSize(resolution_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            background = np.zeros((text_size[1] + 10, text_size[0] + 10, 3), dtype=np.uint8)
            background[:, :] = background_color
            text_position = (10, 30 + text_size[1])
            cv2.putText(background, resolution_text, (10, text_size[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            debug_image[0:text_size[1] + 10, 0:text_size[0] + 10] = background
            debug_image = cv2.resize(debug_image, (image.shape[1] // 2, image.shape[0] // 2))
            filename = os.path.splitext(os.path.basename(image_path))[0]
            debug_image_path = os.path.join(debug_upperbody_folder if option == 1 else debug_face_folder, f"{filename}_face_{i}.jpg")
            cv2.imwrite(debug_image_path, debug_image)
            output_folder = output_upperbody_folder if option == 1 else output_face_folder
            output_image_path = os.path.join(output_folder, f"{filename}_face_{i}.jpg")
            cv2.imwrite(output_image_path, resized_image)
    progress_bar.update(1)
progress_bar.close()
total_images = len(image_paths)
processed_images = total_images - len(os.listdir(obstructed_folder))
skipped_images = len(os.listdir(obstructed_folder))
print(f"Total images: {total_images}")
print(f"Processed images: {processed_images}")
print(f"Skipped images: {skipped_images}")