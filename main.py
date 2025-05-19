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


if __name__ == "__main__":
    cap = cv2.VideoCapture(3)
    cap.set(3, 640)
    cap.set(4, 480)

    if not cap.isOpened():
        print("cannot open camera")
        exit()

    cam = Cam(cap)
    while True:
        frame = cam.read_cam()
        gray = cam.show_cam(frame)
        cam.read_text(gray)
