"""Manual-only Buffalo S webcam smoke test; safe for pytest to import."""

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def main() -> None:
    # Explicit model name and CPU provider avoid FaceAnalysis's buffalo_l default.
    app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    print("Camera started. Press Q to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Could not read frame")
                break

            faces = app.get(frame)
            for face in faces:
                x1, y1, x2, y2 = map(int, face.bbox)
                embedding = face.normed_embedding
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Face | Embedding: {len(embedding)}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                print("Embedding shape:", embedding.shape)
                print("Embedding norm:", np.linalg.norm(embedding))

            cv2.imshow("GymVision AI - InsightFace Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
