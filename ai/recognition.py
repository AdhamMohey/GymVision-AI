"""Local Buffalo S inference from a single webcam frame.

Frames are consumed in memory and never returned or written to disk. Model
initialization is lazy and refuses to download model packs automatically.
"""

from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

BUFFALO_S_MODEL_VERSION = "buffalo_s-512-v1"
CAMERA_INDEX = 1
CAMERA_BACKEND = cv2.CAP_DSHOW
CAMERA_FRAME_TIMEOUT_SECONDS = 5.0


class RecognitionError(RuntimeError):
    """Base error for local recognition and camera failures."""


class RecognitionUnavailable(RecognitionError):
    """The local model pack or inference runtime is unavailable."""


class CameraUnavailable(RecognitionError):
    """The configured local webcam could not be opened or read."""


class NoFaceDetected(RecognitionError):
    """The captured frame contains no detectable face."""


class MultipleFacesDetected(RecognitionError):
    """The captured frame contains more than one detectable face."""


class InvalidEmbedding(RecognitionError):
    """The model did not return a valid 512-dimensional embedding."""


class CameraSessionActive(RecognitionError):
    """Another local preview already owns the webcam."""


class CameraSessionNotActive(RecognitionError):
    """No local preview session is available for capture."""


class CaptureCancelled(RecognitionError):
    """The preview was cancelled while capture was processing."""


def _is_valid_frame(frame: Any) -> bool:
    """Reject empty and all-black startup frames while keeping dark scenes valid."""
    return isinstance(frame, np.ndarray) and frame.size > 0 and bool(np.any(frame))


class _CameraPreviewSession:
    """One camera owner that continuously reads actual frames into memory."""

    def __init__(self, camera: Any) -> None:
        self.camera = camera
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._frame: np.ndarray | None = None
        self._error: str | None = None
        self._released = False
        self._close_lock = threading.Lock()
        self.cancelled = False
        self._worker = threading.Thread(target=self._read_frames, daemon=True)

    def start(self) -> None:
        self._worker.start()

    def _read_frames(self) -> None:
        while not self._stop.is_set():
            try:
                ret, frame = self.camera.read()
            except Exception as exc:  # camera driver errors must end the session cleanly
                with self._condition:
                    self._error = str(exc) or "Camera frame read failed"
                    self._condition.notify_all()
                return
            if not ret or frame is None:
                with self._condition:
                    self._error = "Could not read a frame from the webcam"
                    self._condition.notify_all()
                return
            if not _is_valid_frame(frame):
                time.sleep(0.01)
                continue
            with self._condition:
                self._frame = frame
                self._condition.notify_all()
            time.sleep(0.01)

    def latest_frame(self, timeout: float = 2.0) -> np.ndarray:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._frame is None and self._error is None and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CameraUnavailable("Timed out waiting for a webcam frame")
                self._condition.wait(remaining)
            if self._error:
                raise CameraUnavailable(self._error)
            if self._frame is None:
                raise CameraUnavailable("Camera preview is closed")
            return self._frame.copy()

    def close(self, *, cancelled: bool = False) -> None:
        with self._close_lock:
            self.cancelled = self.cancelled or cancelled
            self._stop.set()
            with self._condition:
                self._condition.notify_all()
            try:
                if not self._released:
                    self._released = True
                    self.camera.release()
            finally:
                if self._worker.is_alive() and threading.current_thread() is not self._worker:
                    self._worker.join(timeout=2.0)


class BuffaloSRecognizer:
    """Capture one frame and return one normalized Buffalo S embedding."""

    def __init__(
        self,
        *,
        model: Any | None = None,
        model_factory: Callable[..., Any] | None = None,
        camera_factory: Callable[..., Any] | None = None,
        model_root: Path | None = None,
        det_size: tuple[int, int] = (640, 640),
        det_threshold: float = 0.50,
    ) -> None:
        self._model = model
        self._model_factory = model_factory
        self._camera_factory = camera_factory or cv2.VideoCapture
        self._model_lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._preview_lock = threading.Lock()
        self._preview_session: _CameraPreviewSession | None = None
        configured_root = os.getenv("GYMVISION_INSIGHTFACE_ROOT")
        self._model_root = model_root or Path(
            configured_root or (Path.home() / ".insightface")
        )
        if len(det_size) != 2 or any(size <= 0 for size in det_size):
            raise ValueError("det_size must contain two positive dimensions")
        if not 0.0 <= det_threshold <= 1.0:
            raise ValueError("det_threshold must be between 0 and 1")
        self._det_size = det_size
        self._det_threshold = det_threshold

    def _get_model(self) -> Any:
        with self._model_lock:
            if self._model is not None:
                return self._model

            pack_dir = self._model_root / "models" / "buffalo_s"
            required_files = ("det_500m.onnx", "w600k_mbf.onnx")
            if not pack_dir.is_dir() or not all((pack_dir / name).is_file() for name in required_files):
                raise RecognitionUnavailable(
                    "Local buffalo_s model files are missing. Install the official pack manually; "
                    "automatic model downloads are disabled."
                )

            factory = self._model_factory
            if factory is None:
                try:
                    from insightface.app import FaceAnalysis
                except ImportError as exc:
                    raise RecognitionUnavailable("InsightFace is not installed in this environment") from exc
                factory = FaceAnalysis

            try:
                self._model = factory(
                    name="buffalo_s",
                    root=str(self._model_root),
                    providers=["CPUExecutionProvider"],
                )
                self._model.prepare(
                    ctx_id=0,
                    det_size=self._det_size,
                    det_thresh=self._det_threshold,
                )
            except Exception as exc:
                self._model = None
                raise RecognitionUnavailable(f"Could not initialize local buffalo_s: {exc}") from exc
            return self._model

    def capture_embedding(self) -> list[float]:
        """Capture one image locally and return only its normalized embedding."""
        model = self._get_model()
        with self._capture_lock:
            camera = self._camera_factory(CAMERA_INDEX, CAMERA_BACKEND)
            session: _CameraPreviewSession | None = None
            try:
                if not camera.isOpened():
                    raise CameraUnavailable(f"Could not open webcam at camera index {CAMERA_INDEX}")
                session = _CameraPreviewSession(camera)
                session.start()
                frame = session.latest_frame(timeout=CAMERA_FRAME_TIMEOUT_SECONDS)
                return self._embedding_from_frame(model, frame)
            finally:
                if session is not None:
                    session.close()
                else:
                    camera.release()

    def _embedding_from_frame(self, model: Any, frame: np.ndarray) -> list[float]:
        faces = model.get(frame)
        if len(faces) == 0:
            raise NoFaceDetected("No face was detected; center one face in the camera view and retry")
        if len(faces) != 1:
            raise MultipleFacesDetected("More than one face was detected; show only one face and retry")
        try:
            embedding = np.asarray(faces[0].normed_embedding, dtype=np.float32)
        except (AttributeError, TypeError, ValueError) as exc:
            raise InvalidEmbedding("The model did not return a numeric normalized embedding") from exc
        if embedding.shape != (512,) or not np.isfinite(embedding).all():
            raise InvalidEmbedding("Expected one finite 512-dimensional embedding")
        norm = float(np.linalg.norm(embedding))
        if not math.isfinite(norm) or norm <= 0:
            raise InvalidEmbedding("The model returned an empty embedding")
        return (embedding / norm).astype(np.float32).tolist()

    def start_preview(self) -> None:
        """Open one local webcam session for preview and later recognition."""
        self._get_model()
        with self._preview_lock:
            if self._preview_session is not None:
                raise CameraSessionActive("A camera preview session is already active")
            camera = self._camera_factory(CAMERA_INDEX, CAMERA_BACKEND)
            try:
                if not camera.isOpened():
                    raise CameraUnavailable(f"Could not open webcam at camera index {CAMERA_INDEX}")
                session = _CameraPreviewSession(camera)
                self._preview_session = session
                session.start()
                session.latest_frame(timeout=CAMERA_FRAME_TIMEOUT_SECONDS)
            except Exception:
                if self._preview_session is not None:
                    self._preview_session.close()
                    self._preview_session = None
                else:
                    camera.release()
                raise

    def preview_jpeg(self) -> bytes:
        """Return a JPEG of the latest live frame; the image stays in memory."""
        with self._preview_lock:
            session = self._preview_session
        if session is None:
            raise CameraSessionNotActive("No camera preview session is active")
        frame = session.latest_frame()
        try:
            encoded, jpeg = cv2.imencode(".jpg", frame)
        except cv2.error as exc:
            raise CameraUnavailable("Could not encode the local camera preview frame") from exc
        if not encoded:
            raise CameraUnavailable("Could not encode the local camera preview frame")
        return jpeg.tobytes()

    def capture_preview_embedding(self) -> list[float]:
        """Recognize the latest preview frame and release its camera in all cases."""
        with self._capture_lock:
            with self._preview_lock:
                session = self._preview_session
            if session is None:
                raise CameraSessionNotActive("Start the camera preview before capturing")
            try:
                frame = session.latest_frame()
                result = self._embedding_from_frame(self._get_model(), frame)
                if session.cancelled:
                    raise CaptureCancelled("Camera capture was cancelled")
                return result
            finally:
                self._close_preview_session(session)

    def cancel_preview(self) -> None:
        """Cancel any active capture and release the webcam immediately."""
        self._close_preview_session(cancelled=True)

    def _close_preview_session(
        self, session: _CameraPreviewSession | None = None, *, cancelled: bool = False
    ) -> None:
        with self._preview_lock:
            active = self._preview_session
            if session is not None and active is not session:
                return
            self._preview_session = None
        if active is not None:
            active.close(cancelled=cancelled)
