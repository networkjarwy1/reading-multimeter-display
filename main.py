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
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

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

class Cam:
    def __init__(self, cap, templates):
        self.cap = cap
        self.templates = templates
        os.makedirs("captured_images", exist_ok=True)
        os.makedirs("samples", exist_ok=True)

        for (root, dirs, file) in os.walk("templates/"):
            for f in file:
                self.templates.append(f)
                print(f"importing {f}")

        self.templates = sorted(self.templates)
        print(self.templates)

        i = 0
        for template in self.templates:
            self.templates[i] = cv2.imread("templates/" + template, cv2.IMREAD_GRAYSCALE)
            i += 1

    def read_cam(self):
        ret, frame = cap.read()
        if not ret:
            print("can't receive frame. exiting...")

        if cv2.waitKey(1) == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            exit()

        return frame

    def show_cam(self, frame):
        cv2.imshow('Camera', frame)

    def read_text(self, gray):
        text = pytesseract.image_to_string(gray)
        print(text)

    def save_frames(self, median, frame, gray):
        cv2.imwrite("samples/" + frame_files[0], frame)
        cv2.imwrite("samples/" + frame_files[1], median)
        cv2.imwrite("samples/" + frame_files[2], gray)

    def match_frames(self):
        frame = cv2.imread("samples/" + frame_files[0])
        median = cv2.imread("samples/" + frame_files[1])
        gray = cv2.imread("samples/" + frame_files[2])

        sample = [frame, median, gray]

        for t in self.templates:
            template = cv2.imread(f'templates/{t}', cv2.IMREAD_GRAYSCALE)
            for i in range(3):
                w, h = template.shape[::-1]

                res = cv2.matchTemplate(sample[i], template[i], cv2.TM_CCOEFF_NORMED)
                threshold = 0.8
                loc = np.where( res >= threshold)
                for pt in zip(*loc[::-1]):
                    cv2.rectangle(sample[i], pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

                cv2.imwrite(f'captured_images/res_{sample[i]}{datetime.now().strftime("%H:%M:%S")}.jpg',sample[i])

    def get_multimeter_zoom(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        rect = find_red_u_shape(frame)
        if rect is not None:
            x, y, w, h = rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            pad = 2
            roi = frame[y + pad:y + h - pad, x + pad:x + w - pad]
            zoom = cv2.resize(roi, (800, 320), interpolation=cv2.INTER_LINEAR) \
                   if roi.size else None
        else:
            zoom = None

        cv2.imshow("Kamera", frame)
        cv2.imshow("Zoom",
                   zoom if zoom is not None else np.zeros((300, 600, 3), np.uint8))

        return zoom


# ---------- main ----------
if __name__ == "__main__":
    cap = cv2.VideoCapture(2)

    templates = []

    if not cap.isOpened():
        print("cannot open camera")
        sys.exit()

    cam   = Cam(cap, templates)

    while cap.isOpened():
        frame = cam.read_cam()
        median = cv2.medianBlur(frame, 5)
        gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
        cam.show_cam(median)
        #cam.read_text(gray)
        cam.save_frames(median, frame, gray)
        cam.match_frames()
