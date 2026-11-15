"""Unit tests for Buffalo S capture without real camera or model weights."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import ai.recognition as recognition
from ai.recognition import (
    BuffaloSRecognizer,
    CameraUnavailable,
    CameraSessionActive,
    CameraSessionNotActive,
    InvalidEmbedding,
    MultipleFacesDetected,
    NoFaceDetected,
    RecognitionUnavailable,
)


class FakeCamera:
    def __init__(self, is_open=True, read_ok=True, frames=None):
        self.open = is_open
        self.read_ok = read_ok
        self.released = False
        self.release_count = 0
        self.frame = np.zeros((8, 8, 3), dtype=np.uint8)
        self.frame[:, :, 1] = 120
        self.frames = list(frames or [])
        self.frame_index = 0

    def isOpened(self):
        return self.open

    def read(self):
        if not self.read_ok:
            return False, None
        if self.frames:
            frame = self.frames[min(self.frame_index, len(self.frames) - 1)]
            self.frame_index += 1
            return True, frame
        return True, self.frame

    def release(self):
        self.released = True
        self.release_count += 1


class FakeModel:
    def __init__(self, faces):
        self.faces = faces
        self.seen_frame = None

    def get(self, frame):
        self.seen_frame = frame
        return self.faces


def face_with_embedding(embedding=None):
    if embedding is None:
        embedding = np.zeros(512, dtype=np.float32)
        embedding[0], embedding[1] = 3.0, 4.0
    return SimpleNamespace(normed_embedding=embedding)


def test_capture_returns_only_normalized_512_embedding_and_releases_camera():
    camera = FakeCamera()
    model = FakeModel([face_with_embedding()])
    arguments = []

    def camera_factory(index, backend):
        arguments.append((index, backend))
        return camera

    result = BuffaloSRecognizer(model=model, camera_factory=camera_factory).capture_embedding()

    assert len(result) == 512
    assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-6)
    assert result[:2] == pytest.approx([0.6, 0.8], abs=1e-6)
    assert arguments == [(1, cv2.CAP_DSHOW)]
    assert camera.released
    assert np.array_equal(model.seen_frame, camera.frame)


@pytest.mark.parametrize(
    ("faces", "error"),
    [([], NoFaceDetected), ([face_with_embedding(), face_with_embedding()], MultipleFacesDetected)],
)
def test_capture_rejects_zero_or_multiple_faces_and_releases_camera(faces, error):
    camera = FakeCamera()

    with pytest.raises(error):
        BuffaloSRecognizer(model=FakeModel(faces), camera_factory=lambda *_: camera).capture_embedding()

    assert camera.released


@pytest.mark.parametrize("opened,read_ok", [(False, True), (True, False)])
def test_capture_reports_camera_failure(opened, read_ok):
    camera = FakeCamera(is_open=opened, read_ok=read_ok)

    with pytest.raises(CameraUnavailable):
        BuffaloSRecognizer(model=FakeModel([]), camera_factory=lambda *_: camera).capture_embedding()

    assert camera.released


def test_model_factory_is_explicit_and_uses_cpu_provider(tmp_path: Path):
    model_dir = tmp_path / "models" / "buffalo_s"
    model_dir.mkdir(parents=True)
    (model_dir / "det_500m.onnx").touch()
    (model_dir / "w600k_mbf.onnx").touch()
    camera = FakeCamera()
    model = FakeModel([face_with_embedding()])
    factory_calls = []
    prepare_calls = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        model.prepare = lambda **arguments: prepare_calls.append(arguments)
        return model

    result = BuffaloSRecognizer(
        model_factory=factory,
        camera_factory=lambda *_: camera,
        model_root=tmp_path,
        det_size=(320, 320),
        det_threshold=0.50,
    ).capture_embedding()

    assert len(result) == 512
    assert factory_calls == [{
        "name": "buffalo_s",
        "root": str(tmp_path),
        "providers": ["CPUExecutionProvider"],
    }]
    assert prepare_calls == [{"ctx_id": 0, "det_size": (320, 320), "det_thresh": 0.50}]
    assert camera.released


def test_missing_model_pack_fails_before_opening_camera(tmp_path: Path):
    camera_calls = []
    recognizer = BuffaloSRecognizer(
        model_factory=lambda **_kwargs: pytest.fail("model factory must not run when pack is missing"),
        camera_factory=lambda *_: camera_calls.append(True),
        model_root=tmp_path,
    )

    with pytest.raises(RecognitionUnavailable, match="automatic model downloads are disabled"):
        recognizer.capture_embedding()

    assert camera_calls == []


def test_wrong_embedding_shape_is_rejected_and_camera_released():
    camera = FakeCamera()
    model = FakeModel([face_with_embedding(np.ones(128, dtype=np.float32))])

    with pytest.raises(InvalidEmbedding, match="512-dimensional"):
        BuffaloSRecognizer(model=model, camera_factory=lambda *_: camera).capture_embedding()

    assert camera.released


def test_capture_waits_past_initial_black_frame_and_releases_camera():
    black_frame = np.zeros((8, 8, 3), dtype=np.uint8)
    camera = FakeCamera(frames=[black_frame, np.full((8, 8, 3), 40, dtype=np.uint8)])
    model = FakeModel([face_with_embedding()])

    def camera_factory(index, backend):
        assert (index, backend) == (1, cv2.CAP_DSHOW)
        return camera

    embedding = BuffaloSRecognizer(
        model=model,
        camera_factory=camera_factory,
    ).capture_embedding()

    assert len(embedding) == 512
    assert np.array_equal(model.seen_frame, camera.frames[1])
    assert camera.released and camera.release_count == 1


def test_capture_times_out_on_black_startup_frames_and_releases_camera(monkeypatch):
    monkeypatch.setattr(recognition, "CAMERA_FRAME_TIMEOUT_SECONDS", 0.05)
    black_frame = np.zeros((8, 8, 3), dtype=np.uint8)
    camera = FakeCamera(frames=[black_frame])

    with pytest.raises(CameraUnavailable, match="Timed out waiting for a webcam frame"):
        BuffaloSRecognizer(
            model=FakeModel([]), camera_factory=lambda *_: camera
        ).capture_embedding()

    assert camera.released and camera.release_count == 1


def test_live_preview_returns_actual_camera_frames_and_capture_releases_camera():
    camera = FakeCamera()
    model = FakeModel([face_with_embedding()])
    arguments = []
    recognizer = BuffaloSRecognizer(
        model=model,
        camera_factory=lambda index, backend: (arguments.append((index, backend)) or camera),
    )

    recognizer.start_preview()
    jpeg = recognizer.preview_jpeg()
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    result = recognizer.capture_preview_embedding()

    assert decoded is not None and decoded.shape == camera.frame.shape
    assert np.allclose(decoded[0, 0], camera.frame[0, 0], atol=5)
    assert len(result) == 512
    assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-6)
    assert arguments == [(1, cv2.CAP_DSHOW)]
    assert model.seen_frame is not None
    assert camera.released and camera.release_count == 1
    with pytest.raises(CameraSessionNotActive):
        recognizer.preview_jpeg()


def test_preview_waits_for_non_black_startup_frame(monkeypatch):
    black_frame = np.zeros((8, 8, 3), dtype=np.uint8)
    camera = FakeCamera(frames=[black_frame, np.full((8, 8, 3), 80, dtype=np.uint8)])
    recognizer = BuffaloSRecognizer(model=FakeModel([]), camera_factory=lambda *_: camera)

    recognizer.start_preview()
    jpeg = recognizer.preview_jpeg()
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert decoded is not None
    assert float(decoded.mean()) > 0
    recognizer.cancel_preview()
    assert camera.released and camera.release_count == 1


def test_preview_start_timeout_releases_camera_after_only_black_frames(monkeypatch):
    monkeypatch.setattr(recognition, "CAMERA_FRAME_TIMEOUT_SECONDS", 0.05)
    black_frame = np.zeros((8, 8, 3), dtype=np.uint8)
    camera = FakeCamera(frames=[black_frame])
    recognizer = BuffaloSRecognizer(model=FakeModel([]), camera_factory=lambda *_: camera)

    with pytest.raises(CameraUnavailable, match="Timed out waiting for a webcam frame"):
        recognizer.start_preview()

    assert camera.released and camera.release_count == 1
    with pytest.raises(CameraSessionNotActive):
        recognizer.preview_jpeg()


def test_preview_prevents_a_second_camera_session():
    camera = FakeCamera()
    recognizer = BuffaloSRecognizer(model=FakeModel([]), camera_factory=lambda *_: camera)
    recognizer.start_preview()

    with pytest.raises(CameraSessionActive):
        recognizer.start_preview()

    recognizer.cancel_preview()
    assert camera.released and camera.release_count == 1


def test_cancelled_preview_releases_camera_and_cannot_be_captured():
    camera = FakeCamera()
    recognizer = BuffaloSRecognizer(model=FakeModel([face_with_embedding()]), camera_factory=lambda *_: camera)
    recognizer.start_preview()
    recognizer.cancel_preview()

    assert camera.released and camera.release_count == 1
    with pytest.raises(CameraSessionNotActive):
        recognizer.capture_preview_embedding()


def test_preview_capture_error_releases_camera():
    camera = FakeCamera()
    recognizer = BuffaloSRecognizer(model=FakeModel([]), camera_factory=lambda *_: camera)
    recognizer.start_preview()

    with pytest.raises(NoFaceDetected):
        recognizer.capture_preview_embedding()

    assert camera.released and camera.release_count == 1
