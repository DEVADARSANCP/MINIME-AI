import cv2
import dlib
import numpy as np
import os

# Initialize dlib's face detector and shape predictor (requires a pre-trained model file)
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure you have this file

def extract_color_features(region):
    """Extract average RGB color from a specified region."""
    avg_color = np.mean(region, axis=(0, 1))[:3]  # RGB channels
    return np.array(avg_color, dtype=int)

def classify_hair_length(forehead, skin_color, threshold=40):
    """Classifies hair length and checks for baldness."""
    # Extract hair color (assumed from the forehead region)
    hair_color = extract_color_features(forehead)

    print(f"Hair Color: {hair_color}, Skin Color: {skin_color}")  # Debug output

    # Check similarity to skin color
    color_difference = np.linalg.norm(hair_color - skin_color)
    print(f"Color Difference: {color_difference}")  # Debug output

    if color_difference < threshold:
        return "Bald"  # Classify as bald if hair color matches skin color

    # Additional logic for hair length classification
    if hair_color[0] < 100:  # Simplistic approach for darker hair
        return "Short Hair"
    else:
        return "Long Hair"

def get_facial_regions(image):
    """Detect facial regions (hair, skin, eyes) using dlib landmarks."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    if not faces:
        print("No face detected")
        return None, None

    # Use the first detected face
    face = faces[0]
    landmarks = shape_predictor(gray, face)

    # Extract regions based on landmarks
    forehead = image[face.top():face.top() + (face.bottom() - face.top()) // 4, face.left():face.right()]
    skin_region = image[landmarks.part(30).y:landmarks.part(33).y, landmarks.part(30).x-10:landmarks.part(30).x+10]

    return forehead, skin_region

def process_image(image_path):
    """Processes the image to extract color features and classify hair length."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None

    # Extract regions
    forehead, skin_region = get_facial_regions(image)
    if forehead is None or skin_region is None:
        print(f"Error: Could not detect required regions in image {image_path}")
        return None

    # Extract color features
    skin_color = extract_color_features(skin_region)

    # Classify hair length
    hair_length = classify_hair_length(forehead, skin_color)

    return {
        "skin_color": skin_color,
        "hair_length": hair_length
    }

# Folder path where face images are stored
folder_path = "./face"

# Process each image in the folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if file_path.endswith(('.jpg', '.jpeg', '.png')):
        results = process_image(file_path)
        if results:
            print(f"Image: {filename}")
            print(f"Skin Color (RGB): {results['skin_color']}")
            print(f"Hair Length: {results['hair_length']}")
            print("\n" + "-"*40 + "\n")
