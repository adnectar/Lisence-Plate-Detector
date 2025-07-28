Automatic License Plate Recognition & Vehicle Tracking with YOLO and OCR

This project implements a complete **Automatic License Plate Recognition (ALPR)** system using deep learning and traditional image processing techniques. It performs **real-time vehicle detection**, **license plate localization**, **text recognition (OCR)**, and **multi-object tracking** on video footage.

Project Overview

The system performs the following tasks frame-by-frame from a video:

1. **Vehicle Detection**
   Uses a YOLOv8 model trained on the COCO dataset to detect vehicles such as:

   * Cars
   * Trucks
   * Buses
   * Motorcycles

2. **License Plate Detection**
   A separate, custom-trained YOLOv8 model is used to detect license plate regions within each frame.

3. **License Plate Recognition (OCR)**

   * Crops the detected license plate region.
   * Applies image preprocessing using OpenCV:

     * Grayscale conversion
     * Thresholding for better contrast
   * Uses EasyOCR to extract the alphanumeric characters from the plate.
   * Post-processes the output to correct common OCR misreads (e.g., `O → 0`, `A → 4`) and validates the format.

4. **Vehicle Tracking**
   Utilizes the **SORT (Simple Online and Realtime Tracking)** algorithm to assign a unique ID to each detected vehicle across multiple frames. This ensures consistent tracking of each vehicle and associates its license plate across time.

5. **Data Export**
   All recognized vehicles and plates are stored per frame and written to a `.csv` file. Each row includes:

   * Frame number
   * Vehicle ID
   * Vehicle bounding box
   * License plate bounding box
   * Detection and OCR confidence scores
   * Final license plate text

Techniques Used

* **Object Detection**: YOLOv8 (`ultralytics`) for vehicle and license plate localization.
* **Optical Character Recognition (OCR)**: EasyOCR for extracting license numbers.
* **Image Preprocessing**: OpenCV techniques for improving OCR accuracy.
* **Multi-Object Tracking**: SORT algorithm for tracking unique vehicles over time.
* **Format Validation & Correction**: Rule-based checks and character mapping for common OCR errors.

Inputs & Outputs

* **Input**: A video file (e.g., `.mp4`) with moving vehicles and visible license plates.
* **Output**: A structured CSV file containing:

  * Tracked vehicle IDs
  * License plate texts
  * Bounding boxes and scores

Possible Use Cases

* Parking management
* Traffic monitoring
* Toll booth automation
* Law enforcement and security analytics

Dependencies

* Python 3.8+
* OpenCV
* Ultralytics (YOLOv8)
* EasyOCR
* NumPy
* SORT tracking module
