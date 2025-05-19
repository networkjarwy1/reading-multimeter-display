import cv2
import pytesseract


class Cam:
    def __init__(self, cap):
        self.cap = cap

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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray

    def read_text(self, gray):
        text = pytesseract.image_to_string(gray)
        print(text)

# ---------- auto-crop displeje ----------
import numpy as np
import time, sys

LOWER_HSV = np.array([15, 30, 40])
UPPER_HSV = np.array([30, 150, 200])
MIN_AREA  = 15000
OUT_W, OUT_H = 600, 300
LOCK_TIME = 1.0

def order_pts(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(1); d = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect

class DisplayDetector:
    def __init__(self):
        self.lock_M = None
        self.lock_start = 0

    def process(self, frame):
        if self.lock_M is not None and (time.time()-self.lock_start) < LOCK_TIME:
            crop = cv2.warpPerspective(frame, self.lock_M, (OUT_W, OUT_H))
            return crop
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8),2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN , np.ones((5,5),np.uint8),2)
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < MIN_AREA:
            return None
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        pts = approx.reshape(-1,2).astype("float32") if len(approx)==4 else cv2.boxPoints(cv2.minAreaRect(c)).astype("float32")
        rect = order_pts(pts)
        dst  = np.float32([[0,0],[OUT_W,0],[OUT_W,OUT_H],[0,OUT_H]])
        M    = cv2.getPerspectiveTransform(rect, dst)
        crop = cv2.warpPerspective(frame, M, (OUT_W, OUT_H))
        self.lock_M     = M
        self.lock_start = time.time()
        return crop

# ---------- main ----------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)      
    cap.set(3, 640)
    cap.set(4, 480)
    if not cap.isOpened():
        print("cannot open camera")
        sys.exit()

    cam   = Cam(cap)
    disp  = DisplayDetector()

    while True:
        frame = cam.read_cam()
        cam.show_cam(frame)                    
        crop = disp.process(frame)            
        if crop is not None:
            cv2.imshow('Display', crop)
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            cam.read_text(gray_crop)           
        else:
            cv2.destroyWindow('Display')
