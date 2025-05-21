import cv2
import pytesseract
import sys, os
import numpy as np
from datetime import datetime

frame_files = ["frame.jpg", "median.jpg", "gray.jpg"]

def find_red_u_shape(frame, min_area=5_000):
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

def extract_display_area(frame):
    """Extract just the display area from the multimeter image"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to isolate the LCD display (which is typically darker)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find largest contour by area (likely the display)
    display_contour = None
    max_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h else 0
        
        # LCD displays typically have aspect ratio > 2
        if area > 5000 and aspect_ratio > 1.5 and aspect_ratio < 5 and y < frame.shape[0] // 2:
            if area > max_area:
                max_area = area
                display_contour = contour
    
    if display_contour is not None:
        x, y, w, h = cv2.boundingRect(display_contour)
        # Expand slightly to ensure we get all digits
        x = max(0, x - 5)
        y = max(0, y - 5)
        w = min(frame.shape[1] - x, w + 10)
        h = min(frame.shape[0] - y, h + 10)
        
        display = frame[y:y+h, x:x+w]
        return display, (x, y, w, h)
    
    return None, None

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


class Cam:
    def __init__(self, cap, templates):
        self.cap = cap
        os.makedirs("captured_images", exist_ok=True)
        os.makedirs("samples", exist_ok=True)
        os.makedirs("display_captures", exist_ok=True)  # New folder for display images

        for root, dirs, files in os.walk("templates/"):
            for f in files:
                if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue                         # vynechat ne-obrázky (.DS_Store apod.)
                templates.append(f)
                print(f"importing {f}")

        self.template_files = sorted(templates)
        print(self.template_files)

        self.template_imgs = [
            cv2.imread(os.path.join("templates", f), cv2.IMREAD_GRAYSCALE)
            for f in self.template_files
        ]
        
        # For stabilizing readings
        self.last_readings = []
        self.stable_reading = None

    def get_multimeter_zoom(self):
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
    
    def process_display(self, zoom):
        """Process the display to read the measurement value"""
        if zoom is None:
            return None, None
        
        # Extract the display area
        display, display_rect = extract_display_area(zoom)
        
        if display is not None and display_rect is not None:
            # Draw rectangle around detected display on the zoom image
            x, y, w, h = display_rect
            cv2.rectangle(zoom, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Preprocess for OCR
            preprocessed = preprocess_display_for_ocr(display)
            
            if preprocessed is not None:
                # Save display captures periodically
                timestamp = datetime.now().strftime('%H-%M-%S')
                cv2.imwrite(os.path.join("display_captures", f"display_{timestamp}.jpg"), display)
                cv2.imwrite(os.path.join("display_captures", f"preproc_{timestamp}.jpg"), preprocessed)
                
                # Display the preprocessed image
                cv2.imshow("Display", display)
                cv2.imshow("Preprocessed", preprocessed)
                
                # Read the display
                text = read_display(preprocessed)
                value, unit = extract_measurement(text)
                
                if value is not None:
                    # Add to sliding window of readings for stability
                    self.last_readings.append((value, unit))
                    if len(self.last_readings) > 5:  # Keep last 5 readings
                        self.last_readings.pop(0)
                    
                    # Only update the stable reading if we have enough consistent values
                    if len(self.last_readings) >= 3:
                        # Get most common reading
                        values = [r[0] for r in self.last_readings]
                        avg_value = sum(values) / len(values)
                        
                        # If all readings are within 5% of the average, consider it stable
                        if all(abs(v - avg_value) / avg_value < 0.05 for v in values):
                            # Get most common unit
                            units = [r[1] for r in self.last_readings]
                            most_common_unit = max(set(units), key=units.count)
                            
                            self.stable_reading = (avg_value, most_common_unit)
                            print(f"Stable reading: {avg_value} {most_common_unit}")
                
                return text, value, unit
        
        return None, None, None

    def save_frames(self, frame, median, gray):
        cv2.imwrite(os.path.join("samples", frame_files[0]), frame)
        cv2.imwrite(os.path.join("samples", frame_files[1]), median)
        cv2.imwrite(os.path.join("samples", frame_files[2]), gray)

    def match_frames(self):
        samples = [
            cv2.imread(os.path.join("samples", f), cv2.IMREAD_GRAYSCALE)
            for f in frame_files
        ]

        for i in range(3):
            img = samples[i]
            if img is None:
                continue

            matched_templates = []
            any_match = False

            for template, tname in zip(self.template_imgs, self.template_files):
                if template is None:
                    continue

                # přeskočit, pokud je šablona větší než vzorek
                if img.shape[0] < template.shape[0] or img.shape[1] < template.shape[1]:
                    continue

                res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= 0.8)

                if loc[0].size:
                    any_match = True
                    matched_templates.append(tname)
                    w, h = template.shape[::-1]
                    for pt in zip(*loc[::-1]):
                        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), 0, 2)

            if any_match:
                print(f"{frame_files[i]} → match: {', '.join(matched_templates)}")
                cv2.imwrite(
                    os.path.join(
                        "captured_images",
                        f"res_{i}_{datetime.now().strftime('%H-%M-%S')}.jpg",
                    ),
                    img,
                )


# ---------- main ----------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    templates = []

    if not cap.isOpened():
        print("cannot open camera")
        sys.exit()

    cam = Cam(cap, templates)

    while cap.isOpened():
        zoom = cam.get_multimeter_zoom()
        if zoom is None:
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        median = cv2.medianBlur(zoom, 5)
        gray   = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)

        # Process the display to read the measurement
        text, value, unit = cam.process_display(zoom)
        if text:
            print(f"Raw Display: {text}")
            if value is not None:
                print(f"Measurement: {value} {unit}")

        cam.save_frames(zoom, median, gray)
        cam.match_frames()

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()