
import cv2

cap = cv2.VideoCapture(1, cv2.CAP_MSMF)

if not cap.isOpened():
    print("Cannot open DroidCam")
    exit()

print("DroidCam connected! Press Q to exit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot receive frame")
        break

    cv2.imshow("GymVision AI - Samsung Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
