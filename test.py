import cv2
import numpy as np


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


def get_multimeter_zoom(cap):
    ret, frame = cap.read()
    if not ret:
        return None

    rect = find_red_u_shape(frame)
    if rect is not None:
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        pad = 2
        roi = frame[y + pad:y + h - pad, x + pad:x + w - pad]
        zoom = cv2.resize(roi, (600, 300), interpolation=cv2.INTER_LINEAR) \
               if roi.size else None
    else:
        zoom = None

    cv2.imshow("Kamera", frame)
    cv2.imshow("Zoom",
               zoom if zoom is not None else np.zeros((300, 600, 3), np.uint8))

    return zoom


def main():
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        raise RuntimeError("Nelze otevřít kameru.")

    cv2.namedWindow("Kamera")
    cv2.namedWindow("Zoom")
    cv2.resizeWindow("Zoom", 600, 300)

    try:
        while True:
            zoom = get_multimeter_zoom(cap)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
