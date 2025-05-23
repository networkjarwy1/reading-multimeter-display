import cv2
import numpy as np
from datetime import datetime
import os
import pytesseract
import time
import sys

possible_units = ['mV', 'V', 'kV', 'MV', 'mA', 'A', 'mΩ', 'Ω', 'kΩ', 'MΩ', 'm Hz', 'Hz', 'k Hz', 'M Hz', 'C', 'F', '.']
show_without_unit = True

filename = datetime.now().strftime("%H:%M:%S_%d.%m.%Y")
txtfile = open(f"results/{filename}.txt", 'w')

def check_unit(unit):
    for i in possible_units:
        if unit == i:
            return unit

    return None     # in case it doesn't match any possible unit

def find_red_u_shape(frame, min_area=5_000):
    """Najde červený U-shape v obraze a vrátí jeho bounding rectangle"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 100, 80), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (160, 100, 80), (180, 255, 255))
    mask  = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h
        if 0.5 < aspect < 4.0 and area > best_area:
            best, best_area = (x, y, w, h), area
    return best

def preprocess_display_for_ocr(display):
    """Preprocess the display image for better OCR results - black/white conversion"""
    if display is None:
        return None

    # Resize for better OCR
    display = cv2.resize(display, (display.shape[1] * 2, display.shape[0] * 2))

    unblured = cv2.medianBlur(display, 5)
    gray = cv2.cvtColor(unblured, cv2.COLOR_BGR2GRAY)
    #bandw = exposure.rescale_intensity(gray, out_range= (0,255))

    # Create black/white image - everything that is black stays black, rest becomes white
    # First invert so black becomes white, then threshold to make it pure black/white
    _ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

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

class Cam:
    def __init__(self, cap):
        self.cap = cap
        os.makedirs("captured_images", exist_ok=True)
        os.makedirs("results", exist_ok=True)


    def get_multimeter_zoom(self):
        """Získá zoom na displej multimetru pomocí detekce červeného U-shape"""
        ret, frame = self.cap.read()
        if not ret:
            return None

        rect = find_red_u_shape(frame)
        if rect is not None:
            x, y, w, h = rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            pad = 2
            roi  = frame[y + pad : y + h - pad, x + pad : x + w - pad]
            zoom = (
                cv2.resize(roi, (800, 320), interpolation=cv2.INTER_LINEAR)
                if roi.size
                else None
            )
        else:
            zoom = None

        cv2.imshow("Kamera", frame)
        cv2.imshow(
            "Zoom", zoom if zoom is not None else np.zeros((300, 600, 3), np.uint8)
        )
        return zoom


# ---------- main ----------
if __name__ == "__main__":
    cap = cv2.VideoCapture(2)  # Změněno z 2 na 0

    if not cap.isOpened():
        print("cannot open camera")
        sys.exit()

    cam = Cam(cap)

    while cap.isOpened():
        # Získání zoom obrazu z multimetru
        zoom = cam.get_multimeter_zoom()
        if zoom is None:
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        # Zpracování pro template matching
        median = cv2.medianBlur(zoom, 5)
        gray   = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)

        # OCR zpracování - black/white konverze a čtení číslic
        bw_image = preprocess_display_for_ocr(zoom)
        if bw_image is not None:
            cv2.imshow('Black/White', bw_image)

            text = read_display(bw_image)
            value, unit = extract_measurement(text)


            if value is not None:
                unit = check_unit(unit)
                print(unit)

                if show_without_unit is True or show_without_unit is False and unit is not None:
                    now = datetime.now()
                    now = now.strftime("%H:%M:%S_%d.%m.%Y")
                    result = f"Value: {value} {unit}     Time: {now}"
                    print(result)
                    cv2.imwrite(f"captured_images/{now}.jpg", zoom)
                    txtfile.write(f'{result}\n')

        if cv2.waitKey(1) == ord('q'):
            break

        time.sleep(0.5)

    txtfile.close()
    cap.release()
    cv2.destroyAllWindows()
