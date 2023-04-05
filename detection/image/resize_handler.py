from typing import Optional

import cv2
from detection.image.image_handler import CV2Image, Dimension
from image_handler import ImageHandler

class ResizeHandler(ImageHandler):
    """Handles image resizing"""

    def __init__(self, size: Dimension, next_handler: Optional['ImageHandler'] = None):
        super().__init__(next_handler)
        self._size = size


    def _handle(self, image: CV2Image) -> None:
        cv2.resize(image, self._size, image, interpolation = cv2.INTER_AREA)