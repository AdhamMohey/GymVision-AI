from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from ai.droidcam_camera import DroidCamPyAVCapture


class FakeContainer:
    def __init__(self):
        self.streams = [SimpleNamespace(type="video", index=0)]
        self.closed = False

    def decode(self, *, video):
        assert video == 0
        return iter([SimpleNamespace(to_ndarray=lambda *, format: np.ones((2, 3, 3), dtype=np.uint8))])

    def close(self):
        self.closed = True


def test_named_pyav_capture_converts_frame_to_bgr_and_releases(monkeypatch):
    container = FakeContainer()
    calls = []

    def open_source(source, **kwargs):
        calls.append((source, kwargs))
        return container

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=open_source))
    camera = DroidCamPyAVCapture()

    opened, frame = camera.read()
    camera.release()

    assert calls == [(
        "video=DroidCam Video",
        {
            "format": "dshow",
            "options": {"video_size": "640x480", "framerate": "30"},
            "timeout": 8,
        },
    )]
    assert camera.isOpened() is False
    assert opened and frame.shape == (2, 3, 3)
    assert np.all(frame == 1)
    assert container.closed


def test_named_pyav_capture_reports_device_open_error(monkeypatch):
    def fail_open(*_args, **_kwargs):
        raise OSError("DirectShow rejected the named device")

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=fail_open))

    import pytest

    with pytest.raises(RuntimeError, match="DroidCam Video.*DirectShow rejected"):
        DroidCamPyAVCapture()
