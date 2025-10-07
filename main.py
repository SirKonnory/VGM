"""VGM prototype desktop application using PySide6."""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import h5py
import numpy as np
from PySide6.QtCore import Qt, QPoint, QRect, QTimer, Signal
from PySide6.QtGui import (QAction, QBrush, QColor, QDragEnterEvent,
                           QDropEvent, QImage, QKeyEvent, QMouseEvent,
                           QPainter, QPen, QPixmap)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                               QDialog, QFileDialog, QFrame, QHBoxLayout,
                               QLabel, QListWidget, QListWidgetItem,
                               QMainWindow, QMessageBox, QPushButton,
                               QSizePolicy, QSlider, QSplitter,
                               QStackedWidget, QToolBar, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class PipelineTree(QTreeWidget):
    """Tree widget that lists pipeline stages."""

    selection_changed = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.populate()
        self.itemSelectionChanged.connect(self._handle_selection_changed)

    def populate(self) -> None:
        stages = [
            ("Работа с видео", 0),
            ("Пересчёт координат", 1),
            ("Визуализация", 2),
            ("Анализ", 3),
        ]
        for title, index in stages:
            item = QTreeWidgetItem([title])
            item.setData(0, Qt.UserRole, index)
            self.addTopLevelItem(item)
        self.setCurrentItem(self.topLevelItem(0))

    def _handle_selection_changed(self) -> None:
        item = self.currentItem()
        if item is None:
            return
        page_index = item.data(0, Qt.UserRole)
        if isinstance(page_index, int):
            logging.debug("Pipeline selection changed to %s", item.text(0))
            self.selection_changed.emit(page_index)


class PlaceholderPage(QWidget):
    """Placeholder widget for future modules."""

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel(f"{title} — в разработке")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)


@dataclass
class VideoMetadata:
    """Container for video metadata."""

    path: str
    width: int
    height: int
    fps: float
    frame_count: int


class FrameView(QLabel):
    """Display widget for video frames with overlay support."""

    clicked = Signal(QPoint)
    file_dropped = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setAcceptDrops(True)
        self._pixmap: Optional[QPixmap] = None
        self._points: List[Tuple[int, float, float]] = []

    def set_frame(self, pixmap: Optional[QPixmap]) -> None:
        self._pixmap = pixmap
        self.update()

    def set_points(self, points: List[Tuple[int, float, float]]) -> None:
        self._points = points
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
            if self._points:
                scale_x = scaled.width() / self._pixmap.width()
                scale_y = scaled.height() / self._pixmap.height()
                for point_id, px, py in self._points:
                    sx = x + int(px * scale_x)
                    sy = y + int(py * scale_y)
                    pen = QPen(QColor("white"))
                    pen.setWidth(3)
                    painter.setPen(pen)
                    painter.setBrush(QBrush(QColor("black")))
                    painter.drawEllipse(QPoint(sx, sy), 6, 6)
                    painter.setPen(QPen(QColor("yellow")))
                    painter.drawText(QPoint(sx + 8, sy - 8), f"{point_id}")
        else:
            painter.drawText(self.rect(), Qt.AlignCenter, "Перетащите видео сюда")

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            pos = event.position().toPoint()
            if self._pixmap:
                scaled = self._pixmap.scaled(
                    self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                x = (self.width() - scaled.width()) // 2
                y = (self.height() - scaled.height()) // 2
                if QRect(x, y, scaled.width(), scaled.height()).contains(pos):
                    rel_x = (pos.x() - x) / scaled.width()
                    rel_y = (pos.y() - y) / scaled.height()
                    px = rel_x * self._pixmap.width()
                    py = rel_y * self._pixmap.height()
                    self.clicked.emit(QPoint(int(px), int(py)))
        super().mousePressEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.isLocalFile():
                self.file_dropped.emit(url.toLocalFile())
                event.acceptProposedAction()
                return
        super().dropEvent(event)


class VideoPage(QWidget):
    """Main page for working with videos and annotations."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self._video_capture: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None
        self._current_index: int = 0
        self._keyframe_index: Optional[int] = None
        self._points: List[Tuple[int, float, float]] = []
        self._frame_request_pending = False

        self._setup_ui()

    # region UI setup
    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # Toolbar
        self.toolbar = QToolBar("Video Toolbar")
        self.open_action = QAction("Открыть видео", self)
        self.open_action.triggered.connect(self._open_video_dialog)
        self.toolbar.addAction(self.open_action)

        self.video_path_label = QLabel("Нет видео")
        self.video_path_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred
        )
        self.toolbar.addWidget(self.video_path_label)

        self.save_action = QAction("Сохранить в HDF5…", self)
        self.save_action.triggered.connect(self._save_to_hdf5)
        self.save_action.setEnabled(False)
        self.toolbar.addAction(self.save_action)

        main_layout.addWidget(self.toolbar)

        # Central area
        center_splitter = QSplitter(Qt.Horizontal)
        self.frame_view = FrameView()
        self.frame_view.clicked.connect(self._handle_frame_click)
        self.frame_view.file_dropped.connect(self.load_video)
        center_splitter.addWidget(self.frame_view)

        self.annotation_panel = self._create_annotation_panel()
        center_splitter.addWidget(self.annotation_panel)
        center_splitter.setStretchFactor(0, 3)
        center_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(center_splitter)

        # Navigation controls
        navigation_layout = QHBoxLayout()
        self.prev_button = QPushButton("⏮ Предыдущий кадр")
        self.next_button = QPushButton("⏭ Следующий кадр")
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.sliderMoved.connect(self._slider_moved)
        self.frame_slider.sliderReleased.connect(self._slider_released)

        self.prev_button.clicked.connect(self._prev_frame)
        self.next_button.clicked.connect(self._next_frame)

        navigation_layout.addWidget(self.prev_button)
        navigation_layout.addWidget(self.frame_slider)
        navigation_layout.addWidget(self.next_button)

        self.frame_label = QLabel("Кадр: -/-")
        navigation_layout.addWidget(self.frame_label)

        self.keyframe_button = QPushButton("Сделать стартовым кадром")
        self.keyframe_button.clicked.connect(self._set_keyframe)
        navigation_layout.addWidget(self.keyframe_button)

        main_layout.addLayout(navigation_layout)

    def _create_annotation_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.add_mode_checkbox = QCheckBox("Режим добавления точек")
        self.add_mode_checkbox.setChecked(True)
        layout.addWidget(self.add_mode_checkbox)

        hint = QLabel("Клик по кадру добавляет ключевую точку")
        layout.addWidget(hint)

        self.points_list = QListWidget()
        self.points_list.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.points_list)

        delete_button = QPushButton("Удалить выбранную точку")
        delete_button.clicked.connect(self._delete_selected_point)
        layout.addWidget(delete_button)

        self.keyframe_status = QLabel("Стартовый кадр не выбран")
        layout.addWidget(self.keyframe_status)

        layout.addStretch(1)
        return panel

    # endregion

    # region Video loading
    def _open_video_dialog(self) -> None:
        dialog = QFileDialog(self, "Открыть видео")
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilters([
            "Видео файлы (*.mp4 *.avi *.mov *.mkv)",
            "Все файлы (*)",
        ])
        if dialog.exec() == QDialog.Accepted:
            selected = dialog.selectedFiles()
            if selected:
                self.load_video(selected[0])

    def load_video(self, path: str) -> None:
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Ошибка", "Выбранный файл недоступен.")
            return
        logging.info("Loading video: %s", path)

        if self._video_capture:
            self._video_capture.release()
            self._video_capture = None

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Ошибка", "Не удалось открыть видео.")
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        self._metadata = VideoMetadata(path, width, height, fps, frame_count)
        self._video_capture = cap
        self._current_index = 0
        self._keyframe_index = None
        self._points.clear()
        self.points_list.clear()
        self.keyframe_status.setText("Стартовый кадр не выбран")
        self.save_action.setEnabled(True)

        self.video_path_label.setText(os.path.basename(path))
        self.video_path_label.setToolTip(path)

        self.frame_slider.setRange(0, max(frame_count - 1, 0))
        self.frame_slider.setEnabled(frame_count > 0)
        self.frame_slider.setValue(0)

        self._request_frame(0)

    # endregion

    # region Frame navigation
    def _request_frame(self, index: int) -> None:
        if not self._video_capture:
            return
        if self._frame_request_pending:
            return
        self._frame_request_pending = True

        def load_frame() -> None:
            pixmap = self._grab_frame(index)
            if pixmap:
                self._current_index = index
                self._update_frame_display(pixmap)
            self._frame_request_pending = False

        QTimer.singleShot(0, load_frame)

    def _grab_frame(self, index: int) -> Optional[QPixmap]:
        if not self._video_capture:
            return None
        cap = self._video_capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            logging.error("Failed to read frame %s", index)
            QMessageBox.critical(
                self, "Ошибка", "Не удалось прочитать кадр из видео."
            )
            return None
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._numpy_to_pixmap(image)

    def _update_frame_display(self, pixmap: QPixmap) -> None:
        self.frame_view.set_frame(pixmap)
        self.frame_view.set_points(self._points)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self._current_index)
        self.frame_slider.blockSignals(False)

        if self._metadata:
            self.frame_label.setText(
                f"Кадр: {self._current_index + 1}/{self._metadata.frame_count}"
            )

    def _slider_moved(self, position: int) -> None:
        self._request_frame(position)

    def _slider_released(self) -> None:
        position = self.frame_slider.value()
        self._request_frame(position)

    def _prev_frame(self) -> None:
        if not self._metadata:
            return
        new_index = max(0, self._current_index - 1)
        self._request_frame(new_index)

    def _next_frame(self) -> None:
        if not self._metadata:
            return
        new_index = min(self._metadata.frame_count - 1, self._current_index + 1)
        self._request_frame(new_index)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() == Qt.Key_Left:
            self._prev_frame()
            event.accept()
            return
        if event.key() == Qt.Key_Right:
            self._next_frame()
            event.accept()
            return
        super().keyPressEvent(event)

    # endregion

    # region Keyframe and points
    def _set_keyframe(self) -> None:
        if not self._metadata:
            return
        self._keyframe_index = self._current_index
        self.keyframe_status.setText(
            f"Стартовый кадр: {self._keyframe_index + 1}"
        )

    def _handle_frame_click(self, point: QPoint) -> None:
        if not self.add_mode_checkbox.isChecked():
            return
        if self._metadata is None:
            return
        if self._keyframe_index is None:
            QMessageBox.information(
                self,
                "Стартовый кадр",
                "Сначала выберите стартовый кадр, затем размечайте точки.",
            )
            return
        if self._current_index != self._keyframe_index:
            QMessageBox.warning(
                self,
                "Неверный кадр",
                "Разметку можно добавлять только на стартовом кадре.",
            )
            return
        point_id = len(self._points)
        self._points.append((point_id, float(point.x()), float(point.y())))
        self._refresh_points_list()
        self.frame_view.set_points(self._points)

    def _refresh_points_list(self) -> None:
        self.points_list.clear()
        for point_id, px, py in self._points:
            item = QListWidgetItem(f"{point_id}: ({px:.1f}, {py:.1f})")
            item.setData(Qt.UserRole, point_id)
            self.points_list.addItem(item)

    def _delete_selected_point(self) -> None:
        selected = self.points_list.currentItem()
        if not selected:
            return
        point_id = selected.data(Qt.UserRole)
        self._points = [p for p in self._points if p[0] != point_id]
        # Reindex points
        for idx, entry in enumerate(self._points):
            self._points[idx] = (idx, entry[1], entry[2])
        self._refresh_points_list()
        self.frame_view.set_points(self._points)

    # endregion

    # region Saving
    def _save_to_hdf5(self) -> None:
        if self._metadata is None:
            return
        dialog = QFileDialog(self, "Сохранить в HDF5")
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setNameFilters(["HDF5 файлы (*.h5)", "Все файлы (*)"])
        dialog.setDefaultSuffix("h5")
        if dialog.exec() != QDialog.Accepted:
            return
        selected = dialog.selectedFiles()
        if not selected:
            return
        path = selected[0]
        if os.path.exists(path):
            answer = QMessageBox.question(
                self,
                "Подтверждение",
                "Файл существует. Перезаписать?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        keypoints = np.array([[px, py] for _, px, py in self._points], dtype=np.float32)
        if keypoints.size == 0:
            QMessageBox.information(
                self,
                "Предупреждение",
                "Вы сохраняете проект без ключевых точек.",
            )

        tracking = {
            "keyframe_index": self._keyframe_index if self._keyframe_index is not None else -1,
            "keypoints": keypoints,
        }
        try:
            H5Writer.save_project(path, self._metadata, tracking)
        except Exception as exc:  # pragma: no cover - UI message
            logging.exception("Failed to save HDF5")
            QMessageBox.critical(
                self,
                "Ошибка",
                f"Не удалось сохранить файл: {exc}",
            )
            return
        QMessageBox.information(self, "Готово", "Проект успешно сохранён.")

    # endregion

    @staticmethod
    def _numpy_to_pixmap(image: np.ndarray) -> QPixmap:
        height, width, channel = image.shape
        bytes_per_line = channel * width
        q_image = QImage(
            image.data, width, height, bytes_per_line, QImage.Format_RGB888
        )
        return QPixmap.fromImage(q_image.copy())


class H5Writer:
    """Utility class for writing project data to HDF5."""

    @staticmethod
    def save_project(path: str, meta: VideoMetadata, tracking: dict) -> None:
        logging.info("Saving project to %s", path)
        with h5py.File(path, "w") as h5:
            project_group = h5.create_group("project")
            project_group.attrs["version"] = "0.1"
            project_group.attrs["created"] = datetime.now().isoformat()
            project_group.attrs["notes"] = ""

            video_group = h5.create_group("video")
            source_group = video_group.create_group("source")
            source_group.attrs["path"] = meta.path
            source_group.attrs["width"] = meta.width
            source_group.attrs["height"] = meta.height
            source_group.attrs["fps"] = float(meta.fps)
            source_group.attrs["frame_count"] = int(meta.frame_count)

            tracking_group = h5.create_group("tracking")
            keyframe_index = tracking.get("keyframe_index", -1)
            tracking_group.create_dataset(
                "keyframe_index", data=np.int64(keyframe_index)
            )
            keypoints = tracking.get("keypoints")
            if keypoints is None or keypoints.size == 0:
                tracking_group.create_dataset("keypoints", data=np.zeros((0, 2), dtype=np.float32))
            else:
                tracking_group.create_dataset("keypoints", data=keypoints.astype(np.float32))


class MainWindow(QMainWindow):
    """Main application window with pipeline navigation."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Видеограмметрия (ВГМ)")
        self.resize(1280, 720)

        splitter = QSplitter()
        self.setCentralWidget(splitter)

        self.pipeline_tree = PipelineTree()
        splitter.addWidget(self.pipeline_tree)

        self.pages = QStackedWidget()
        splitter.addWidget(self.pages)

        self.video_page = VideoPage()
        self.pages.addWidget(self.video_page)
        self.pages.addWidget(PlaceholderPage("Пересчёт координат"))
        self.pages.addWidget(PlaceholderPage("3D визуализация"))
        self.pages.addWidget(PlaceholderPage("Анализ"))

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

        self.pipeline_tree.selection_changed.connect(self._change_page)

    def _change_page(self, index: int) -> None:
        if 0 <= index < self.pages.count():
            self.pages.setCurrentIndex(index)


def main(argv: List[str]) -> int:
    """Application entry point."""

    app = QApplication(argv)
    window = MainWindow()
    window.show()

    if len(argv) > 1:
        video_path = argv[1]
        window.video_page.load_video(video_path)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
