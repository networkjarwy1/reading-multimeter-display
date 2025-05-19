import cv2

cap = cv2.VideoCapture(3)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    print("cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("can't receive frame. exiting...")
        break

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
