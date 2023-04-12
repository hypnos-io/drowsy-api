import cv2 as cv

from types import Optional
from image_handler import ImageHandler
from drowsiness.drowsy_types import CV2Image, Dimension


class ResizeHandler(ImageHandler):
    """Handles image resizing"""

    def __init__(self, size: Dimension, next_handler: Optional["ImageHandler"] = None):
        super().__init__(next_handler)
        self._size = size

    def _handle(self, image: CV2Image) -> None:
        cv.resize(image, self._size, image, interpolation=cv.INTER_AREA)
