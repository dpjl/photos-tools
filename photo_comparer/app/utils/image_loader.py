"""Asynchronous image loader using Qt's global thread pool.

Threading guarantee
-------------------
Each _LoadTask creates its own _Signals QObject **on the GUI thread**
(inside load(), which is always called from the GUI thread).  The task
runs in a pool worker and emits _Signals.loaded.  Because _Signals lives
on the GUI thread, Qt automatically uses a QueuedConnection and delivers
the emission on the GUI thread.

ImageLoader re-exposes a single image_loaded signal so callers can connect
to one place with a proper QObject slot — guaranteeing GUI-thread delivery
even if connected after load() is called.
"""
from pathlib import Path

from PIL import Image as PILImage
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtGui import QImage


class _Signals(QObject):
    loaded = Signal(str, QImage)
    error  = Signal(str, str)


class _LoadTask(QRunnable):
    def __init__(self, key: str, path: Path):
        super().__init__()
        self.key = key
        self.path = path
        self.signals = _Signals()   # created on GUI thread → queued delivery
        self.setAutoDelete(True)

    def run(self):
        image = QImage(str(self.path))
        if image.isNull():
            # Fallback: Pillow handles formats that Qt may lack on Linux (e.g. TIFF)
            try:
                with PILImage.open(self.path) as pil_img:
                    pil_img = pil_img.convert("RGBA")
                    data = pil_img.tobytes("raw", "RGBA")
                    image = QImage(
                        data,
                        pil_img.width,
                        pil_img.height,
                        pil_img.width * 4,
                        QImage.Format.Format_RGBA8888,
                    ).copy()  # .copy() détache du buffer local `data`
            except Exception:
                pass
        if image.isNull():
            self.signals.error.emit(self.key, f"Cannot load {self.path.name}")
        else:
            self.signals.loaded.emit(self.key, image)


class ImageLoader(QObject):
    """Delivers image_loaded **always on the GUI thread** via signal→signal."""

    image_loaded = Signal(str, QImage)  # key, image  — GUI thread
    load_error   = Signal(str, str)     # key, message — GUI thread

    def __init__(self, parent: QObject = None):
        super().__init__(parent)
        self._pool = QThreadPool.globalInstance()

    def load(self, key: str, path: Path):
        task = _LoadTask(key, path)
        # signal→signal: Qt queues cross-thread emission automatically
        task.signals.loaded.connect(self.image_loaded)
        task.signals.error.connect(self.load_error)
        self._pool.start(task)
