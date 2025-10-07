"""Entry point for launching the VGM desktop application."""
from __future__ import annotations

import logging
import sys
from typing import List

from PySide6.QtWidgets import QApplication

from vgm.frontend import MainWindow


def configure_logging() -> None:
    """Configure logging for console output."""

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def main(argv: List[str]) -> int:
    """Start the Qt event loop and show the main window."""

    configure_logging()

    app = QApplication(argv)
    window = MainWindow()
    window.show()

    if len(argv) > 1:
        window.video_page.load_video(argv[1])

    return app.exec()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
