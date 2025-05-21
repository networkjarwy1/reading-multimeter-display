import cv2
import sys, os
import numpy as np
from datetime import datetime

frame_files = ["frame.jpg", "median.jpg", "gray.jpg"]


class Cam:
    def __init__(self, cap, templates):
        self.cap = cap
        os.makedirs("captured_images", exist_ok=True)
        os.makedirs("sample", exist_ok=True)

    def get_multimeter_zoom(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                return None

            cv2.imshow('Camera', frame)

            cv2.imwrite('sample/image.jpg', frame)
            image = cv2.imread('sample/image.jpg')

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            lower_red = np.array([0, 100, 50])
            upper_red = np.array([10, 255, 255])

            mask = cv2.inRange(hsv, lower_red, upper_red)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            u_shape_contour = None
            for contour in contours:
                if len(contour) > 10:
                    u_shape_contour = contour
                    break

            if np.any(u_shape_contour != None):
                break

        print("u_shape_contour: ", u_shape_contour)
        epsilon = 0.02 * cv2.arcLength(u_shape_contour, True)
        approx_contour = cv2.approxPolyDP(u_shape_contour, epsilon, True)

        # Ensure we only have four points (corners of the U-shape)
        if len(approx_contour) > 4:
            # If more than four points are detected, simplify further
            epsilon = max(0.001 * cv2.arcLength(approx_contour, True), 5)  # Example adjustment
            approx_contour = cv2.approxPolyDP(approx_contour, epsilon, True)

        # Print the four corner points of the U-shape
        print("Four corner points of the U-shape:")
        print(approx_contour)

        # Visualize the detected contour and its corners on the image
        i = 0
        points = [[0, 0], [0, 0], [0, 0], [0, 0]]
        for point in approx_contour:
            x, y = point[0]
            points[i] = [x.tolist(), y.tolist()]
            cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
            i += 1
            if i >= 4:
                print("points:")
                print(points)
                break

        pts1 = np.float32(points)
        pts2 = np.float32([[0,0],[0,320],[800,320],[800,0]])


        M = cv2.getPerspectiveTransform(pts1,pts2)

        dst = cv2.warpPerspective(image,M,(800,320))

        cv2.imshow('U-shaped Object', dst)

        return dst


# ---------- main ----------
if __name__ == "__main__":
    cap = cv2.VideoCapture(2)
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

    cap.release()
    cv2.destroyAllWindows()
