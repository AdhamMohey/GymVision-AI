import cv2


def main() -> None:
    camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("ERROR: Camera could not be opened.")
        return

    print("GymVision AI Camera Started!")
    print("Press Q to quit.")

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to capture frame.")
                break

            cv2.putText(
                frame,
                "GymVision AI - Camera Test",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("GymVision AI", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
