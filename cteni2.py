import cv2
import pytesseract
import sys, os
import numpy as np
from datetime import datetime
from collections import Counter

# --- Globální konfigurace ---
# Cesta k Tesseractu (pokud není v PATH, odkomentujte a upravte)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Příklad pro Windows

FRAME_FILES_FOR_TEMPLATE_MATCHING = ["frame.jpg", "median.jpg", "gray.jpg"] # Pokud používáte template matching

# --- Detekce multimetru (červený tvar) ---
def find_red_u_shape(frame, min_area=5_000):
    if frame is None: return None
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
        if area < min_area: continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h) if h > 0 else 0
        if 0.5 < aspect < 4.0 and area > best_area:
            best, best_area = (x, y, w, h), area
    return best

# --- Extrakce oblasti displeje ---
def extract_display_area(frame_roi):
    if frame_roi is None:
        # print("DEBUG (ExtractDisplay): Input frame_roi is None.") # Zakomentováno
        return None, None
    
    gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    _, thresh_display_detect = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV) 
    
    kernel_display = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    thresh_display_detect = cv2.morphologyEx(thresh_display_detect, cv2.MORPH_CLOSE, kernel_display, iterations=2)
    thresh_display_detect = cv2.morphologyEx(thresh_display_detect, cv2.MORPH_OPEN, kernel_display, iterations=1)

    contours, _ = cv2.findContours(thresh_display_detect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    display_contour = None
    max_area = 0
    roi_h, roi_w = frame_roi.shape[:2]

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h else 0
        
        min_display_area = roi_w * roi_h * 0.05 
        if area > min_display_area and 1.5 < aspect_ratio < 5.0 and y < roi_h * 0.7: 
            if area > max_area:
                max_area = area
                display_contour = contour
    
    if display_contour is not None:
        x, y, w, h = cv2.boundingRect(display_contour)
        padding_x = int(w * 0.03)
        padding_y = int(h * 0.05)
        
        x_final = max(0, x - padding_x)
        y_final = max(0, y - padding_y)
        w_final = min(roi_w - x_final, w + 2 * padding_x)
        h_final = min(roi_h - y_final, h + 2 * padding_y)
        
        display_img = frame_roi[y_final:y_final+h_final, x_final:x_final+w_final]
        return display_img, (x_final, y_final, w_final, h_final)
    
    # print("DEBUG (ExtractDisplay): Display contour not found in ROI.") # Zakomentováno
    return None, None

# --- ZJEDNODUŠENÉ Předzpracování displeje pro OCR ---
def preprocess_display_for_ocr(display_img):
    if display_img is None:
        return None 

    target_h = 100 
    orig_h, orig_w = display_img.shape[:2]
    if orig_h == 0 or orig_w == 0:
        return None
        
    scale_ratio = target_h / float(orig_h)
    target_w = int(orig_w * scale_ratio)
    if target_w <= 0: target_w = int(target_h * 2.5) # Fallback
    
    try:
        resized_display = cv2.resize(display_img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    except cv2.error:
        return None

    gray = cv2.cvtColor(resized_display, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)

    # --- KLÍČOVÁ ČÁST: Prahování ---
    threshold_value = 110  # <<< !!! ZMĚŇTE TUTO HODNOTU PRO LADĚNÍ !!! >>>
    
    _, temp_mask_inv = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV)
    processed_image = cv2.bitwise_not(temp_mask_inv)

    # print(f"DEBUG (Preproc): Applied threshold_value: {threshold_value}. Mean intensity of processed: {np.mean(processed_image):.2f}") # Zakomentováno

    return processed_image

# --- Čtení displeje pomocí Tesseractu ---
def read_display(preprocessed_img):
    if preprocessed_img is None: return None
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.-+mVACFkΩ%'
    try:
        text = pytesseract.image_to_string(preprocessed_img, config=custom_config)
        return text.strip().replace(" ","")
    except pytesseract.TesseractNotFoundError:
        print("ERROR: Tesseract is not installed or not in your PATH.")
        print("Please install Tesseract OCR and ensure tesseract.exe (or tesseract) is accessible.")
        print("You might need to set pytesseract.pytesseract.tesseract_cmd manually.")
        sys.exit(1)
    except Exception as e:
        # print(f"Error during Tesseract OCR: {e}") # Zakomentováno pro méně výpisů
        return None


# --- Extrakce hodnoty a jednotky ---
def extract_measurement(text):
    if not text: return None, None
    cleaned_text = ''.join(c for c in text if c.isdigit() or c in '.-+mVACFkΩ%')
    number_part, unit_part = '', ''
    first_unit_char_index = -1
    for i, char in enumerate(cleaned_text):
        if char.isalpha() or char in 'Ω%µ':
            first_unit_char_index = i
            break
    if first_unit_char_index != -1:
        number_part = cleaned_text[:first_unit_char_index]
        unit_part = cleaned_text[first_unit_char_index:]
    else:
        number_part = cleaned_text
    if number_part.count('.') > 1:
        parts = number_part.split('.', 1)
        number_part = parts[0] + '.' + parts[1].replace('.', '')
    if number_part.count('-') > 1 or (number_part.count('-') == 1 and not number_part.startswith('-')):
        number_part = number_part.replace('-', '')
    if not number_part or number_part in ['.', '-', '+.', '.+','+']: return None, unit_part.strip()
    try:
        unit_part = unit_part.replace('uV', 'µV').replace('pV', 'µV') 
        unit_part = unit_part.replace('Ao', 'AC').replace('Do', 'DC')
        value = float(number_part)
        return value, unit_part.strip()
    except ValueError:
        if number_part.endswith('.'):
            try: return float(number_part[:-1]), unit_part.strip()
            except ValueError: pass
        return None, unit_part.strip()


# --- Třída kamery a zpracování ---
class Cam:
    def __init__(self, cap):
        self.cap = cap
        os.makedirs("display_captures", exist_ok=True)
        self.last_readings = []
        self.stable_reading_value = None
        self.stable_reading_unit = ""
        self.reading_history_size = 7 
        self.stability_threshold_count = 4 

    def get_multimeter_zoom(self):
        ret, frame = self.cap.read()
        if not ret: 
            # print("WARN: Cannot read frame from camera.") # Zakomentováno
            return None
        cv2.imshow("Kamera", frame)
        rect = find_red_u_shape(frame)
        if rect is not None:
            x, y, w, h = rect
            pad = 2
            roi = frame[y + pad : y + h - pad, x + pad : x + w - pad]
            if roi.size > 0:
                zoom = cv2.resize(roi, (800, 320), interpolation=cv2.INTER_LINEAR)
                return zoom
        # else:
            # print("DEBUG (Zoom): Red U-shape not found.") # Zakomentováno
        return None
    
    def process_display(self, zoom_roi):
        if zoom_roi is None:
            return None, self.stable_reading_value, self.stable_reading_unit

        display_img, display_rect_coords = extract_display_area(zoom_roi)
        
        display_area_to_show = np.zeros((100, 200, 3), dtype=np.uint8)
        if display_img is not None:
            h_disp, w_disp = display_img.shape[:2]
            if h_disp > 0 and w_disp > 0:
                if h_disp < 100 or w_disp < 150 :
                    scale_disp = min(2.0, 150.0/w_disp if w_disp > 0 else 2.0, 100.0/h_disp if h_disp > 0 else 2.0)
                    display_area_to_show = cv2.resize(display_img, (int(w_disp*scale_disp), int(h_disp*scale_disp)), interpolation=cv2.INTER_NEAREST)
                else:
                    display_area_to_show = display_img
            if display_rect_coords is not None:
                dx, dy, dw, dh = display_rect_coords
                cv2.rectangle(zoom_roi, (dx, dy), (dx + dw, dy + dh), (255, 0, 0), 2)
        cv2.imshow("Display Area", display_area_to_show)

        preprocessed_for_ocr = preprocess_display_for_ocr(display_img)
        
        preprocessed_display_show = np.zeros((100 * 3, int(100*2.5*3) , 1), dtype=np.uint8) 
        if preprocessed_for_ocr is not None:
            display_scale_factor = 3
            h_pre, w_pre = preprocessed_for_ocr.shape[:2]
            if h_pre > 0 and w_pre > 0:
                preprocessed_display_show = cv2.resize(preprocessed_for_ocr, 
                                                  (w_pre * display_scale_factor, h_pre * display_scale_factor), 
                                                  interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Preprocessed for OCR", preprocessed_display_show)

        if preprocessed_for_ocr is not None:
            if np.random.rand() < 0.01: # Ukládat jen velmi zřídka
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                if display_img is not None:
                     cv2.imwrite(os.path.join("display_captures", f"display_in_{timestamp}.png"), display_img)
                cv2.imwrite(os.path.join("display_captures", f"preproc_ocr_{timestamp}.png"), preprocessed_for_ocr)
            
            text = read_display(preprocessed_for_ocr)
            value, unit = extract_measurement(text)
            
            if value is not None:
                self.last_readings.append((value, unit if unit else ""))
                if len(self.last_readings) > self.reading_history_size: self.last_readings.pop(0)
                if len(self.last_readings) >= self.stability_threshold_count:
                    values_hist = [r[0] for r in self.last_readings]
                    units_hist = [r[1] for r in self.last_readings]
                    rounded_values = [round(v, 2) for v in values_hist]
                    value_counts = Counter(rounded_values)
                    most_common_val_tuple = value_counts.most_common(1)
                    if most_common_val_tuple:
                        most_common_val, count = most_common_val_tuple[0]
                        if count >= self.stability_threshold_count:
                            self.stable_reading_value = most_common_val
                            relevant_units = [u for val_r, u in zip(rounded_values, units_hist) if val_r == most_common_val]
                            if relevant_units:
                                unit_counts = Counter(relevant_units)
                                self.stable_reading_unit = unit_counts.most_common(1)[0][0]
                            else:
                                self.stable_reading_unit = ""
            return text, value, self.stable_reading_unit # Vracíme aktuální value, a stabilizovanou jednotku
        
        return None, None, self.stable_reading_unit

# --- Hlavní program ---
if __name__ == "__main__":
    # Příklad cesty k Tesseractu (pokud není v PATH, odkomentujte a upravte)
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("CHYBA: Nelze otevřít kameru nebo video soubor.")
        sys.exit()

    cam_obj = Cam(cap)

    zoom_placeholder = np.zeros((300, 600, 3), dtype=np.uint8)
    display_area_placeholder = np.zeros((100, 150, 3), dtype=np.uint8)
    preprocessed_placeholder = np.zeros((100 * 3, int(100*2.5*3) , 1), dtype=np.uint8) 

    print("Spouštění smyčky zpracování. Stiskněte 'q' pro ukončení.")
    print(f"Pro ladění OCR upravte 'threshold_value' ve funkci 'preprocess_display_for_ocr'.")

    while cap.isOpened():
        multimeter_roi = cam_obj.get_multimeter_zoom()

        if multimeter_roi is not None:
            cv2.imshow("Zoom", multimeter_roi)
        else:
            cv2.imshow("Zoom", zoom_placeholder)

        if multimeter_roi is not None:
            # Zde předáváme multimeter_roi do process_display
            # process_display vrátí aktuálně přečtenou hodnotu a stabilizovanou jednotku
            raw_ocr_text, current_parsed_value, stable_unit = cam_obj.process_display(multimeter_roi)
            
            # Vypisujeme pouze pokud je aktuálně parsovaná hodnota platná
            if current_parsed_value is not None:
                # Použijeme stabilizovanou jednotku, pokud je dostupná, jinak prázdný string
                unit_to_display = cam_obj.stable_reading_unit if cam_obj.stable_reading_unit else ""
                print(f"Hodnota: {current_parsed_value} {unit_to_display} (Raw OCR: '{raw_ocr_text}')")

        else:
            cv2.imshow("Display Area", display_area_placeholder)
            cv2.imshow("Preprocessed for OCR", preprocessed_placeholder)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("Ukončování programu...")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Program ukončen.")