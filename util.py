# Import standard libraries
import string
import easyocr

# ===================== OCR SETUP =====================

# Initialize the EasyOCR reader for English (disable GPU for portability)
reader = easyocr.Reader(['en'], gpu=False)

# ===================== CHARACTER MAPPING =====================

# Map commonly misread characters (letter → number)
dict_char_to_int = {
    'O': '0',
    'I': '1',
    'J': '3',
    'A': '4',
    'G': '6',
    'S': '5'
}

# Map commonly misread digits (number → letter)
dict_int_to_char = {
    '0': 'O',
    '1': 'I',
    '3': 'J',
    '4': 'A',
    '6': 'G',
    '5': 'S'
}

# ===================== VEHICLE ASSOCIATION =====================

def get_car(license_plate, vehicle_track_ids):
    """
    Associates a license plate bounding box to the correct vehicle bounding box by checking
    if the plate lies within the vehicle's bounding box.

    Args:
        license_plate (tuple): Bounding box of license plate [x1, y1, x2, y2, score, class_id].
        vehicle_track_ids (list): List of vehicle bounding boxes + tracking IDs:
                                  [x1, y1, x2, y2, track_id].

    Returns:
        tuple: Matched vehicle's bounding box + tracking ID, or all -1s if not found.
    """
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        # Check if license plate bbox lies completely inside the car bbox
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]  # Return matched car box and ID

    return -1, -1, -1, -1, -1  # No match found

# ===================== FORMAT VALIDATION =====================

def license_complies_format(text):
    """
    Check if a license plate string matches a specific 7-character format:
    2 letters, 2 numbers, then 3 letters.

    Args:
        text (str): Raw license plate text from OCR.

    Returns:
        bool: Whether the string complies with the expected format.
    """
    if len(text) != 7:
        return False

    # Check character-by-character with allowance for OCR errors
    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in string.digits or text[2] in dict_char_to_int.keys()) and \
       (text[3] in string.digits or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True

    return False

# ===================== FORMATTING CORRECTIONS =====================

def format_license(text):
    """
    Correct OCR misreads by converting visually similar characters based on position.

    Args:
        text (str): Raw OCR text.

    Returns:
        str: Corrected/cleaned license plate text.
    """
    license_plate_ = ''

    # Define expected position mapping:
    # positions 0,1,4,5,6 → likely letters (use int→char)
    # positions 2,3 → likely numbers (use char→int)
    mapping = {
        0: dict_int_to_char,
        1: dict_int_to_char,
        4: dict_int_to_char,
        5: dict_int_to_char,
        6: dict_int_to_char,
        2: dict_char_to_int,
        3: dict_char_to_int
    }

    for j in range(7):
        if text[j] in mapping[j]:
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_

# ===================== OCR PROCESSING =====================

def read_license_plate(license_plate_crop):
    """
    Extract text from a cropped license plate image using EasyOCR and validate/format it.

    Args:
        license_plate_crop (ndarray): Cropped grayscale/binary image of license plate.

    Returns:
        tuple: (Formatted text, confidence score) or (None, None) if invalid.
    """
    detections = reader.readtext(license_plate_crop)  # OCR on image

    for detection in detections:
        bbox, text, score = detection  # Get text and its OCR confidence
        text = text.upper().replace(' ', '')  # Normalize text

        # Validate format and apply formatting fixes
        if license_complies_format(text):
            return format_license(text), score

    return None, None  # Return nothing if no valid license plate found

# ===================== RESULT EXPORT =====================

def write_csv(results, output_path):
    """
    Export all frame-by-frame license plate recognition results to a CSV file.

    Args:
        results (dict): Nested dictionary of results:
                        {frame_number: {car_id: {car, license_plate}}}
        output_path (str): Path to save the CSV.
    """
    with open(output_path, 'w') as f:
        # Write header
        f.write('{},{},{},{},{},{},{}\n'.format(
            'frame_nmr', 'car_id', 'car_bbox',
            'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
            'license_number_score'))

        # Iterate through results
        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])  # Debug print (can be removed)

                # Make sure we have all required keys to write a valid row
                if 'car' in results[frame_nmr][car_id] and \
                   'license_plate' in results[frame_nmr][car_id] and \
                   'text' in results[frame_nmr][car_id]['license_plate']:

                    f.write('{},{},{},{},{},{},{}\n'.format(
                        frame_nmr,
                        car_id,
                        '[{} {} {} {}]'.format(*results[frame_nmr][car_id]['car']['bbox']),
                        '[{} {} {} {}]'.format(*results[frame_nmr][car_id]['license_plate']['bbox']),
                        results[frame_nmr][car_id]['license_plate']['bbox_score'],
                        results[frame_nmr][car_id]['license_plate']['text'],
                        results[frame_nmr][car_id]['license_plate']['text_score']))
        f.close()
