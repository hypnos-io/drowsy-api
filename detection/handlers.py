"""
Image Handlers
"""

from abc import ABC, abstractmethod
from typing import Optional, Sequence

import cv2 as cv
import numpy as np


class ImageHandler(ABC):
    """Abstract Handler class implementing the Chain-of-responsability pattern"""

    def __init__(self, next_handler: Optional["ImageHandler"] = None):
        self._next = next_handler

    @abstractmethod
    def _handle(self, image: np.ndarray) -> None:
        """Abstract method to be overriden with image processing steps"""
        pass

    def handle(self, image: np.ndarray) -> None:
        """Process image and pass it on to the next handler on the chain"""
        self._handle(image)

        if self._next:
            self._next.handle(image)

    def set_next(self, next_handler: "ImageHandler"):
        """Set the next handler on the chain"""
        self._next = next_handler
        return self


class CropHandler(ImageHandler):
    """Handles image cropping"""

    def _handle(self, image: np.ndarray, ) -> None:
        raise NotImplementedError


class ResizeHandler(ImageHandler):
    """Handles image resizing"""

    def __init__(self, size: Sequence[int], next_handler: Optional["ImageHandler"] = None):
        super().__init__(next_handler)
        self._size = size

    def _handle(self, image: Sequence[int]) -> None:
        cv.resize(image, self._size, image, interpolation=cv.INTER_AREA)
