# Import necessary libraries
from ultralytics import YOLO  # YOLOv8 object detection framework
import cv2  # OpenCV library for video and image processing
from util import get_car, read_license_plate, write_csv  # Custom utility functions
from sort.sort import *  # SORT tracker for multi-object tracking

# ======================= MODEL SETUP =======================

# Load the general COCO-pretrained YOLOv8 model (for vehicle detection)
coco_model = YOLO('yolov8n.pt')  # yolov8n.pt is the "nano" version (lightweight, faster)

# Load the custom-trained license plate detection model
license_plate_detector = YOLO('/Users/dhanvin/Documents/Image Recognition Python/lpr_best.pt')

# ======================= VIDEO SETUP =======================

# Open a video file using OpenCV
cap = cv2.VideoCapture('/Users/dhanvin/Documents/Image Recognition Python/sample.mp4')

# Frame counter (used to index results per frame)
frame_nmr = -1

# List of vehicle class IDs from COCO dataset:
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
vehicles = [2, 3, 5, 7]

# Dictionary to store final results per frame and per vehicle
results = {}

# Initialize the SORT tracker for tracking objects across frames
mot_tracker = Sort()

# Flag for reading frames
ret = True

# ======================= MAIN PROCESSING LOOP =======================

while ret:
    frame_nmr += 1  # Increment frame counter
    ret, frame = cap.read()  # Read the next frame from the video

    if ret:
        # Initialize an empty dictionary for this frame's detections
        results[frame_nmr] = {}

        # Run vehicle detection on the current frame using YOLO
        detections = coco_model(frame)[0]  # Only take first result from YOLO output
        detections_ = []  # This will store filtered vehicle detections

        # Loop through all detected objects in the frame
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection  # Bounding box and class
            if int(class_id) in vehicles:  # Filter to include only vehicles
                # Store [x1, y1, x2, y2, confidence score]
                detections_.append([x1, y1, x2, y2, score])

        # Update SORT tracker with current frame's vehicle detections
        track_ids = mot_tracker.update(np.asarray(detections_))  # Returns [x1, y1, x2, y2, ID]

        # Run license plate detection on the same frame
        license_plates = license_plate_detector(frame)[0]  # Get detection results

        # Process each detected license plate
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate  # Bounding box and score

            # Try to associate this license plate with a tracked vehicle using overlap
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            # Proceed only if a matching vehicle was found
            if car_id != -1:
                # Crop the license plate region from the frame
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # Convert cropped plate to grayscale
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)

                # Apply binary inverse thresholding for OCR preprocessing
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Run OCR to extract text from the license plate image
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                # Store the result only if text was successfully detected
                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {
                        'car': {
                            'bbox': [xcar1, ycar1, xcar2, ycar2]  # Vehicle bounding box
                        },
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],  # License plate bounding box
                            'text': license_plate_text,  # Detected plate text
                            'bbox_score': score,  # Confidence of detection
                            'text_score': license_plate_text_score  # Confidence of OCR
                        }
                    }

# ======================= EXPORT RESULTS =======================

# Write all collected results into a CSV file
write_csv(results, './test.csv')
