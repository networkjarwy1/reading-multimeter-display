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
        cv2.imshow("Camera", frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray

    def read_text(self, gray):
        text = pytesseract.image_to_string(gray)
        print(text)

import numpy as np
import types

points = []        
transform_ready = False
M = None           
w = h = 0            
main_window_closed = False  

def order_points(pts: np.ndarray) -> np.ndarray:
    """Seřadí 4 body do pořadí: [tl, tr, br, bl]."""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]      
    ordered[2] = pts[np.argmax(s)]       
    ordered[1] = pts[np.argmin(diff)]   
    ordered[3] = pts[np.argmax(diff)]    
    return ordered

def mouse_callback(event, x, y, flags, param):
    global points, transform_ready
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        transform_ready = False         

cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", mouse_callback)

def patched_show_cam(self, frame):
    """
    Rozšířená verze show_cam:
      • umožní vybrat 4 body a poté „zamkne“ náhled jen na výřez
      • vrací gray-scale výřezu (je-li aktivní) nebo celé scény
    """
    global points, transform_ready, M, w, h, main_window_closed

    if not main_window_closed:
        for p in points:
            cv2.circle(frame, tuple(p), 5, (0, 255, 0), -1)
        cv2.imshow("Camera", frame)

    if len(points) == 4 and not transform_ready:
        src = np.array(points, dtype="float32")
        ordered = order_points(src)

        (tl, tr, br, bl) = ordered
        widthA  = np.linalg.norm(br - bl)
        widthB  = np.linalg.norm(tr - tl)
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        w = int(max(widthA, widthB))
        h = int(max(heightA, heightB))

        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                       dtype="float32")
        M = cv2.getPerspectiveTransform(ordered, dst)
        transform_ready = True

        cv2.destroyWindow("Camera")
        main_window_closed = True

    roi_gray = None
    if transform_ready and M is not None:
        warped = cv2.warpPerspective(frame, M, (w, h))
        cv2.imshow("vyrez", warped)
        roi_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):       
        points.clear()
        transform_ready = False
        if cv2.getWindowProperty("vyrez", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow("vyrez")
        if main_window_closed:
            cv2.namedWindow("Camera")
            cv2.setMouseCallback("Camera", mouse_callback)
            main_window_closed = False

    if roi_gray is not None:
        return roi_gray
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)    
    cap.set(3, 640)
    cap.set(4, 480)

    if not cap.isOpened():
        print("cannot open camera")
        exit()

    cam = Cam(cap)

    cam.show_cam = types.MethodType(patched_show_cam, cam)

    while True:
        frame = cam.read_cam()
        gray = cam.show_cam(frame)
        cam.read_text(gray)