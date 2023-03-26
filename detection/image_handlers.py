"""
Image Handlers
"""

from abc import ABC, abstractmethod
from typing import Self, Optional, Sequence

import cv2
import numpy as np

CV2Image = np.ndarray
Dimension = Sequence[int]

class ImageHandler(ABC):
    """Abstract Handler class implementing the Chain-of-responsability pattern"""

    def __init__(self, next_handler: Optional['ImageHandler'] = None):
        self._next = next_handler

    @abstractmethod
    def _handle(self, image: CV2Image) -> None:
        """Abstract method to be overriden with image processing steps"""
        pass

    def handle(self, image: CV2Image) -> None:
        """Process image and pass it on to the next handler on the chain"""
        self._handle(image)

        if self._next:
            self._next.handle(image)

    def set_next(self, next_handler: 'ImageHandler') -> Self:
        """Set the next handler on the chain"""
        self._next = next_handler
        return self


class CropHandler(ImageHandler):
    """Handles image cropping"""

    def _handle(self, image: CV2Image) -> None:
        raise NotImplementedError


class ResizeHandler(ImageHandler):
    """Handles image resizing"""

    def __init__(self, size: Dimension, next_handler: Optional['ImageHandler'] = None):
        super().__init__(next_handler)
        self._size = size


    def _handle(self, image: CV2Image) -> None:
        cv2.resize(image, self._size, image, interpolation = cv2.INTER_AREA)

class ColorHandler(ImageHandler):
    """Handles image colorspace conversion"""

    def _handle(self, image: CV2Image) -> None:
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY, image)
