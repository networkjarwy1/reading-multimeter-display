import cv2
import numpy as np
from datetime import datetime
import os
from platform import release

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

        cv2.imshow('Original Image', cv2.imread('sample/image.jpg'))

        if result is not None:
            cv2.imshow('Warped U-Shape', result)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
