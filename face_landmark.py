import cv2
import dlib

def calculate_face_quality(image, startX, startY, endX, endY, is_error):
    x = 0
    y = 0
    quality = 0
    shape = None

    if is_error == False:
        # Initialize the face detector and shape predictor
        shape_predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat") # type: ignore

        face_width = endX - startX

        print(f"face_width: {face_width}")

        face = image[startY:endY, startX:endX]

        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        shape = shape_predictor(gray_face, dlib.rectangle(0, 0, gray_face.shape[1], gray_face.shape[0])) # type: ignore

        # Calculate the face quality based on landmarks
        quality = calculate_quality_metric(shape, face_width)

        quality = min(quality, 100)
        quality = round(quality)

        print(f"startX: {startX}")
        print(f"startY: {startY}")
        print(f"startY: {startY}")
        print(f"endY: {endY}")

    return image, x, y, quality, shape


def calculate_quality_metric(shape, face_width):
    # Calculate the coverage of facial landmarks
    num_landmarks = shape.num_parts
    coverage = num_landmarks / 5.0  # Assuming the shape model has 5 landmarks

    # Calculate the average distance between eye landmarks
    left_eye_coords = [(shape.part(i).x, shape.part(i).y) for i in range(2)]
    right_eye_coords = [(shape.part(i).x, shape.part(i).y) for i in range(2, 4)]
    avg_eye_distance = calculate_avg_euclidean_distance(left_eye_coords, right_eye_coords)

    # Calculate the ratio of eye distance to face width
    eye_distance_ratio = avg_eye_distance / face_width

    # Normalize the metrics to range 0-1
    max_eye_distance_ratio = 0.5  # Define the maximum possible value for eye distance ratio

    normalized_eye_distance_ratio = eye_distance_ratio / max_eye_distance_ratio

    # Adjust the metric weights
    eye_distance_weight = 1.0

    # Calculate the overall face quality based on different factors
    quality = coverage * 100 + normalized_eye_distance_ratio * eye_distance_weight * 100

    quality = min(quality, 100)  # Ensure the quality value is within 0-100 range

    print(quality)
    return quality


def calculate_avg_euclidean_distance(coords1, coords2):
    total_distance = 0
    for i in range(len(coords1)):
        distance = calculate_euclidean_distance(coords1[i], coords2[i])
        total_distance += distance
    avg_distance = total_distance / len(coords1)
    return avg_distance


def calculate_euclidean_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
