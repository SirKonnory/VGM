"""Backend logic for the VGM desktop prototype."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import h5py
import numpy as np


@dataclass
class VideoMetadata:
    """Container for basic video metadata."""

    path: str
    width: int
    height: int
    fps: float
    frame_count: int


class VideoController:
    """Encapsulates video access and annotation state."""

    def __init__(self) -> None:
        self._capture: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None
        self._keyframe_index: Optional[int] = None
        self._points: List[Tuple[int, float, float]] = []

    # region lifecycle -------------------------------------------------
    def load_video(self, path: str) -> VideoMetadata:
        """Load a video file and initialise metadata."""

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Видео не найдено: {path}")

        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise ValueError("Не удалось открыть видео")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)

        self.release()

        self._capture = capture
        self._metadata = VideoMetadata(path, width, height, fps, frame_count)
        self._keyframe_index = None
        self._points.clear()
        logging.info(
            "Video loaded: path=%s size=%sx%s frames=%s fps=%.3f",
            path,
            width,
            height,
            frame_count,
            fps,
        )
        return self._metadata

    def release(self) -> None:
        """Release the current capture if it exists."""

        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self._metadata = None
        self._keyframe_index = None
        self._points.clear()

    def __del__(self) -> None:  # pragma: no cover - defensive clean-up
        try:
            self.release()
        except Exception:  # pragma: no cover - best effort
            logging.exception("Failed to release video controller")

    # endregion --------------------------------------------------------

    # region frame access ----------------------------------------------
    def get_frame(self, index: int) -> np.ndarray:
        """Return the RGB frame at ``index``."""

        if self._capture is None or self._metadata is None:
            raise RuntimeError("Видео не загружено")
        if not 0 <= index < self._metadata.frame_count:
            raise IndexError("Запрошен кадр вне диапазона")

        self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Не удалось прочитать кадр {index}")

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # endregion --------------------------------------------------------

    # region annotation state -----------------------------------------
    @property
    def metadata(self) -> Optional[VideoMetadata]:
        """Return the metadata of the loaded video, if any."""

        return self._metadata

    @property
    def keyframe_index(self) -> Optional[int]:
        return self._keyframe_index

    def set_keyframe(self, index: int) -> None:
        if self._metadata is None:
            raise RuntimeError("Нельзя выбрать стартовый кадр без видео")
        if not 0 <= index < self._metadata.frame_count:
            raise IndexError("Стартовый кадр вне диапазона")
        self._keyframe_index = index
        logging.info("Keyframe selected: %s", index)

    def add_point(self, x: float, y: float) -> Tuple[int, float, float]:
        if self._keyframe_index is None:
            raise RuntimeError("Стартовый кадр не выбран")
        point_id = len(self._points)
        point = (point_id, float(x), float(y))
        self._points.append(point)
        logging.debug("Point added: id=%s x=%.2f y=%.2f", point_id, x, y)
        return point

    def remove_point(self, point_id: int) -> bool:
        filtered = [point for point in self._points if point[0] != point_id]
        if len(filtered) == len(self._points):
            return False
        self._points = [
            (new_id, px, py) for new_id, (_, px, py) in enumerate(filtered)
        ]
        logging.debug("Point removed: id=%s", point_id)
        return True

    def clear_points(self) -> None:
        self._points.clear()

    @property
    def points(self) -> List[Tuple[int, float, float]]:
        return list(self._points)

    def tracking_payload(self) -> dict:
        keypoints = np.array([[px, py] for _, px, py in self._points], dtype=np.float32)
        return {
            "keyframe_index": self._keyframe_index if self._keyframe_index is not None else -1,
            "keypoints": keypoints,
        }

    # endregion --------------------------------------------------------


class H5Writer:
    """Utility helper for storing project data in HDF5 format."""

    @staticmethod
    def save_project(path: str, meta: VideoMetadata, tracking: dict) -> None:
        logging.info("Saving project to %s", path)
        with h5py.File(path, "w") as handle:
            project_group = handle.create_group("project")
            project_group.attrs["version"] = "0.1"
            project_group.attrs["created"] = datetime.now().isoformat()
            project_group.attrs["notes"] = ""

            video_group = handle.create_group("video")
            source_group = video_group.create_group("source")
            source_group.attrs["path"] = meta.path
            source_group.attrs["width"] = meta.width
            source_group.attrs["height"] = meta.height
            source_group.attrs["fps"] = float(meta.fps)
            source_group.attrs["frame_count"] = int(meta.frame_count)

            tracking_group = handle.create_group("tracking")
            keyframe_index = np.int64(tracking.get("keyframe_index", -1))
            tracking_group.create_dataset("keyframe_index", data=keyframe_index)

            keypoints = tracking.get("keypoints")
            if keypoints is None or getattr(keypoints, "size", 0) == 0:
                data = np.zeros((0, 2), dtype=np.float32)
            else:
                data = np.asarray(keypoints, dtype=np.float32)
            tracking_group.create_dataset("keypoints", data=data)
