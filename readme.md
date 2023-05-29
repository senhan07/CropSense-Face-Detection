# CropSense - Face Detection

It is a script to crop an images based on various bounding box sizes, such as upper body, face, or full body. It utilizes a powerful pre-trained s3fd face detection model to accurately identify faces within the images and performs precise cropping based on the detected regions into 1:1 ratio

It proves to be particularly useful in scenarios where you need make a dataset for training a model, it's not perfect yet :)


## Key Features:
- Flexible Crop Options: The script provides three different crop options: upper body, face, and full body. This flexibility allows you to choose the specific region you want to focus on, depending on your requirements.

- Highly Accurate Face Detection: By utilizing the pre-trained s3fd face detection model, the Image Cropper script achieves reliable and accurate face detection. It can handle various face angles, lighting conditions, and occlusions, ensuring that the cropping is performed precisely around the detected faces.

- Efficient Batch Processing: Whether you have a few images or a large collection, the script is designed to process them efficiently. By automating the cropping process, you can save significant time and effort compared to manual cropping.

- Visual Preview Option: The script offers the option to display a preview window, allowing you to visualize the detected bounding boxes before the cropping takes place. This feature gives you a chance to verify and adjust the crop regions, ensuring the desired areas are selected accurately.

- Debugging and Error Handling: The Image Cropper script includes debugging capabilities by saving debug images with bounding boxes. This helps in assessing the accuracy of the face detection and crop regions. Additionally, any images that couldn't be processed or had a low confidence level are saved separately in an error folder for further analysis.

- Customizable Parameters: The script provides configurable parameters to adjust various settings, including output resolution, minimum bounding box sizes for different regions, confidence level for face detection, and folder paths for input and output directories. This allows you to fine-tune the cropping process according to your specific needs.

## Prerequisites

- Python 3.x (Tested on Python 3.10)
- OpenCV (cv2)
- numpy
- tqdm
- pywin32 (win32com.client)

## Installation

1. Clone this repository:
`git clone https://github.com/senhan07/image-cropper.git`


2. Install the required packages:
`pip install -r requirements.txt`


## Usage

1. Place your input images in the `input` folder.
2. Run the script: `python main.py`

3. Follow the prompts to select the crop type (1 for upper body, 2 for face, 3 for full body) and choose whether to show a preview window.
4. If the output folders are not empty, you will be prompted to clean the output folder. Choose accordingly.
5. The script will process each image, perform face detection, and crop the images based on the selected crop type and confidence level.
6. The cropped images will be saved in the respective output folders (`output/upperbody_cropped`, `output/face_cropped`, `output/fullbody_cropped`).
7. Debug images with bounding boxes will be saved in the debug folders (`output/upperbody_debug`, `output/face_debug`, `output/fullbody_debug`).
8. Any images that couldn't be processed or had low confidence level will be created as shorcut in the `output/error_images` folder.
9. The script will display progress bars and provide updates during the processing.

## Configuration

You can customize the following settings in the script:

- `output_res`: Output cropped resolution (pixel).
- `preview_output_res`: Preview window output resolution (pixel).
- `preview_debug_res`: Preview window debug image resolution (pixel).
- `min_face_res`: Minimum detected face size for face crop type (pixel).
- `min_upperbody_res`: Minimum detected face size for upperbody crop type (pixel).
- `min_fullbody_res`: Minimum detected face size for fullbody crop type (pixel).
- `confidence_level`: Confidence level for face detection (between 0 and 1).
- `input_folder`: Input images folder.
- `output_upperbody_folder`: Output folder for upper body cropped images.
- `debug_upperbody_folder`: Debug folder for upper body images with bounding boxes.
- `output_face_folder`: Output folder for face cropped images.
- `debug_face_folder`: Debug folder for face images with bounding boxes.
- `output_fullbody_folder`: Output folder for full body cropped images.
- `debug_fullbody_folder`: Debug folder for full body images with bounding boxes.
- `error_folder`: Folder for error images.
- `cv2.waitKey(250)` Delay time for displaying preview image in (milliseconds)

## License

This script is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Created with my fingers and ChatGPT
- The s3fd face detection model used in this script is based on the [S3FD.pytorch](https://github.com/polarisZhao/S3FD.pytorch) repository.
- Progress bar implementation using [tqdm](https://github.com/tqdm/tqdm).
- Image processing using [OpenCV](https://github.com/opencv/opencv).
