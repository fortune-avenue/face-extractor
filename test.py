import cv2
import os
import requests
from ultralytics import YOLO

# Load YOLO model from a local file
model = YOLO('last.pt')

def extract_faces(image_path, output_folder='extracted_faces'):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")

    # Perform face detection
    results = model(img)

    # Extract bounding boxes
    boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
    confidences = results[0].boxes.conf.cpu().numpy()  # confidence
    classes = results[0].boxes.cls.cpu().numpy()  # class
    face_count = 0
    extracted_faces = []

    image_base_name = os.path.splitext(os.path.basename(image_path))[0]

    for box, conf, cls in zip(boxes, confidences, classes):
        x1, y1, x2, y2 = box
        if int(cls) == 0:  # Class 0 corresponds to 'person' in YOLO
            face_count += 1
            face = img[int(y1):int(y2), int(x1):int(x2)]

            # Save the face image
            face_filename = os.path.join(output_folder, f'{image_base_name}_{face_count}.jpg')
            cv2.imwrite(face_filename, face)
            extracted_faces.append(face_filename)
            print(f"Face {face_count} extracted and saved to {face_filename}")

    if face_count == 0:
        print("No faces found in the image.")
    else:
        print(f"Total {face_count} faces extracted.")
    
    return extracted_faces

def call_face_ktp_matcher_api(img1_path, img2_path, api_url):
    with open(img1_path, 'rb') as img1_file, open(img2_path, 'rb') as img2_file:
        files = {
            'image1': img1_file,
            'image2': img2_file
        }
        response = requests.post(api_url, files=files)
        return response.json()

if __name__ == '__main__':
    ktp_image_path = 'images/dio&ktpnya.jpg'

    # Extract faces from the images
    extracted_faces_ktp = extract_faces(ktp_image_path)

    # Ensure there are extracted faces in both images
    if not extracted_faces_ktp:
        print("No faces extracted from one or both images.")
        exit()

    # Call the face_ktp_matcher API using the first extracted face from each image
    api_url = 'http://127.0.0.1:8000/verification/compare-faces'
    result = call_face_ktp_matcher_api(extracted_faces_ktp[0], extracted_faces_ktp[1], api_url)

    # Print the API result
    print(result)
