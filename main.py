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


class Cam:
    def __init__(self, cap, templates):
        self.cap = cap
        os.makedirs("captured_images", exist_ok=True)
        os.makedirs("samples", exist_ok=True)

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

        cam.save_frames(zoom, median, gray)
        cam.match_frames()

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
