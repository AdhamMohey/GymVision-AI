"""PyAV capture adapter for the explicitly named local DroidCam device."""

from __future__ import annotations

from typing import Any


class DroidCamPyAVCapture:
    """Expose PyAV DirectShow frames through the recognizer's camera interface."""

    device_name = "DroidCam Video"

    def __init__(self) -> None:
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("PyAV is required for the DroidCam camera source") from exc

        try:
            self._container = av.open(
                f"video={self.device_name}",
                format="dshow",
                options={"video_size": "640x480", "framerate": "30"},
                timeout=8,
            )
            self._stream = next(
                stream for stream in self._container.streams if stream.type == "video"
            )
            self._frames = iter(self._container.decode(video=self._stream.index))
        except Exception as exc:
            container = getattr(self, "_container", None)
            if container is not None:
                container.close()
            raise RuntimeError(
                f"Could not open DirectShow device {self.device_name!r}: {exc}"
            ) from exc
        self._released = False

    def isOpened(self) -> bool:
        return not self._released

    def read(self) -> tuple[bool, Any | None]:
        if self._released:
            return False, None
        try:
            frame = next(self._frames)
            return True, frame.to_ndarray(format="bgr24")
        except (StopIteration, EOFError):
            return False, None
        except Exception as exc:
            raise RuntimeError(f"Could not retrieve a frame from {self.device_name!r}: {exc}") from exc

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._container.close()


def create_droidcam_capture(_index: int, _backend: int) -> DroidCamPyAVCapture:
    """Adapt the legacy recognizer camera factory signature to named PyAV input."""
    return DroidCamPyAVCapture()
