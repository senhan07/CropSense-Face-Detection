# CropSense - Face Detection
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/F1F2K9CHH)
<a href="https://www.buymeacoffee.com/_ramen_"><img src="https://img.buymeacoffee.com/button-api/?text=Buy me a coffee&emoji=&slug=_ramen_&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff" height="30px"/></a>

![GitHub repo size](https://img.shields.io/github/repo-size/senhan07/CropSense-Face-Detection?label=SIZE&style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/senhan07/CropSense-Face-Detection?color=red&label=Issues&style=flat-square)
![GitHub pull requests](https://img.shields.io/github/issues-pr/senhan07/CropSense-Face-Detection?label=Pull%20Requests&style=flat-square)
<br>
CropSense is a tool to crop an images based on various bounding box sizes, such as upper body, face, or full body. It utilizes a powerful pre-trained s3fd face detection model to accurately identify faces within the images and performs precise cropping into 1:1 ratio based on the detected regions

It  useful in scenarios where you need make a dataset for training a model, it's not perfect yet :)

- [Demo](#demo)
- [Processing Speed](#processing-speed)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)

### Demo
![Demo](demo.gif "Demo")

## Processing Speed
Currently, the code uses CPU instead of GPU for processing.<br>
1000 images (1440px) take about 1 minute to complete on the Intel i3-12100F 
<br>Running on multithread requires disabling the preview window

## Features:
- 🖼️ **Flexible Crop Options**: Choose from three crop options: upper body, face, or full body. Select the perfect region you want to highlight and let the script do the magic!

- 🔍 **Accurate Face Detection:** By utilizing the pre-trained s3fd face detection model, the script achieves reliable and accurate face detection. It can handle various face angles, lighting conditions, and occlusions, ensuring that the cropping is performed precisely around the detected faces.

- ⏱️ **Effortless Batch Processing:** Whether you have a few images or a whole bunch, this script is designed to handle them all in a jiffy. Say goodbye to manual cropping and hello to lightning-fast automation!

- 🐞 **Debugging Made Easy:** With built-in debugging capabilities, the script saves debug images with bounding boxes, making it a breeze to troubleshoot and ensure your crops are picture-perfect. Any images that couldn't be processed are saved separately for your convenience.

- 🎛️ **Customize to Your Liking:** Tweak the script to match your preferences! Adjust output resolution, minimum bounding box sizes, confidence levels, and folder paths. Make it your own and create stunning visuals!

## Prerequisites

- Python 3.8+ (Tested on Python 3.10)
- OpenCV (cv2)
- numpy
- tqdm
- pywin32 (win32com.client)

## Installation

1. Clone this repository:
`git clone https://github.com/senhan07/CropSense-Face-Detection`

2. Navigate to the project directory:
`cd CropSense-Face-Detection`

3. Install the required packages:
`pip install -r requirements.txt`

## Usage

1. Place your images in the `input` folder
2. Run the script: `python main.py`
3. Follow the prompts to select the crop type (1 for upper body, 2 for face, 3 for full body) and choose whether to show a preview window.
4. If the output folders are not empty, you will be prompted to clean the output folder. Choose accordingly.
5. The script will process each image, perform face detection, and crop the images based on the selected crop type and confidence level.
6. The cropped images will be saved in the respective output folders (`output/upperbody_cropped`, `output/face_cropped`, `output/fullbody_cropped`).
7. Debug images with bounding boxes will be saved in the debug folders (`output/upperbody_debug`, `output/face_debug`, `output/fullbody_debug`).
8. Any images that couldn't be processed or had low confidence level will be created as shorcut in the `output/error_images` folder.

## Configuration

You can customize the following settings in the `variable.py`:

- `output_res`: Output cropped resolution (pixel).
- `preview_output_res`: Preview window output size (pixel).
- `preview_debug_max_res`: Maximum preview window debug image size (pixel).
- `min_face_res`: Minimum detected face size for face crop type (pixel).
- `min_upperbody_res`: Minimum detected face size for upperbody crop type (pixel).
- `min_fullbody_res`: Minimum detected face size for fullbody crop type (pixel).
- `confidence_level`: Confidence level for face detection (between 0 and 1). *(default: 0.5)*
- `input_folder`: Input images folder.
- `output_upperbody_folder`: Output folder for upper body cropped images.
- `debug_upperbody_folder`: Debug folder for upper body images with bounding boxes.
- `output_face_folder`: Output folder for face cropped images.
- `debug_face_folder`: Debug folder for face images with bounding boxes.
- `output_fullbody_folder`: Output folder for full body cropped images.
- `debug_fullbody_folder`: Debug folder for full body images with bounding boxes.
- `error_folder`: Folder for error images.

## Contributing
Contributions to the project are welcome. If you find any issues or would like to suggest improvements, please create a new issue or submit a pull request.

## License

This script is licensed under the [MIT License](LICENSE).
