import cv2
import numpy as np
from datetime import datetime
import os
import pytesseract
import time

def preprocess_display_for_ocr(display):
    """Preprocess the display image for better OCR results"""
    if display is None:
        return None

    # Resize for better OCR
    display = cv2.resize(display, (display.shape[1] * 2, display.shape[0] * 2))

    # Convert to grayscale
    gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)

    # Apply threshold to isolate digits
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Apply some noise reduction
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return thresh

def read_display(preprocessed_img):
    """Read text from preprocessed display image"""
    if preprocessed_img is None:
        return None

    # Configure Tesseract for digits
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.mVACFΩ'

    # Read display text
    text = pytesseract.image_to_string(preprocessed_img, config=custom_config)

    # Clean up the text
    text = text.strip()

    return text

def extract_measurement(text):
    """Extract numerical measurement and unit from OCR text"""
    if not text:
        return None, None

    # Remove any unwanted characters
    cleaned = ''.join(c for c in text if c.isdigit() or c in '.-mVACFΩ')

    # Extract number and unit
    number = ''
    unit = ''
    for c in cleaned:
        if c.isdigit() or c in '.-':
            number += c
        else:
            unit += c

    try:
        value = float(number)
        return value, unit
    except ValueError:
        return None, None

def extract_and_warp_u_shape():
    image = cv2.imread('sample/image.jpg')

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the red color range in HSV (adjust these values according to your image)
    lower_red = np.array([0, 100, 50])
    upper_red = np.array([10, 255, 255])

    # Threshold the image to get only red pixels
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours in the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the U-shape contour
    u_shape_contour = None
    for contour in contours:
        if len(contour) > 10:  # Filter out small noise
            u_shape_contour = contour
            break

    # If no suitable contour found, return None
    if u_shape_contour is None:
        return None

    # Approximate the contour to get four points
    epsilon = 0.02 * cv2.arcLength(u_shape_contour, True)
    approx_contour = cv2.approxPolyDP(u_shape_contour, epsilon, True)

    # Ensure we only have four points (corners of the U-shape)
    if len(approx_contour) > 4:
        # If more than four points are detected, simplify further
        epsilon = max(0.001 * cv2.arcLength(approx_contour, True), 5)  # Example adjustment
        approx_contour = cv2.approxPolyDP(u_shape_contour, epsilon, True)

    # Extract up to four corner points
    points = [[0, 0], [0, 0], [0, 0], [0, 0]]
    i = 0
    for point in approx_contour:
        x, y = point[0]
        points[i] = [x, y]
        cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
        i += 1
        if i >= 4:
            break

    # If we didn't get 4 points, return None
    if i < 4:
        return None

    cv2.imshow('indicator', image)

    # Perform perspective transform
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[0,320],[800,320],[800,0]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image, M, (800, 320))

    return dst

class Cam:
    def __init__(self, cap):
        self.cap = cap
        os.makedirs("captured_images", exist_ok=True)
        os.makedirs("sample", exist_ok=True)

    def read_cam(self):
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite("sample/image.jpg", frame)
            return frame


# ---------- main ----------
if __name__ == "__main__":
    cap = cv2.VideoCapture(2)
    cam = Cam(cap)

    while cap.isOpened:
        cam.read_cam()
        result = extract_and_warp_u_shape()
        prep = preprocess_display_for_ocr(result)
        text = read_display(prep)
        value, unit = extract_measurement(text)

        print(value, " ", unit)

        cv2.imshow('Original Image', cv2.imread('sample/image.jpg'))

        if result is not None:
            cv2.imshow('Warped U-Shape', result)
        if cv2.waitKey(1) == ord('q'):
            break

        time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()
