
import cv2

CAMERA_INDEX = 1

# Load pre-trained Haar Cascade face detector
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Could not load face detection model")

camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not camera.isOpened():
    raise RuntimeError("Could not open camera")

print("GymVision AI - Face Detection Started")
print("Press Q to quit")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face Detected",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("GymVision AI - Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
