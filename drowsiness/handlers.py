"""
Image Handlers
"""

from abc import ABC, abstractmethod
from typing import Optional

import cv2 as cv
import numpy as np

point = tuple[int, int]
dimension = point


class ImageHandler(ABC):
    """Abstract Handler class implementing the Chain-of-responsability pattern"""

    def __init__(
        self,
        next_handler: Optional["ImageHandler"] = None,
        *handler_chain: list["ImageHandler"]
    ):
        self._next = (
            next_handler if not callable(next_handler) else next_handler(*handler_chain)
        )

    @abstractmethod
    def _handle(self, image: np.ndarray) -> None:
        """Abstract method to be overriden with image processing steps"""
        pass

    def handle(self, image: np.ndarray, **kwargs) -> None:
        """Process image and pass it on to the next handler on the chain"""
        self._handle(image, **kwargs)

        if self._next:
            self._next.handle(image, **kwargs)

    def set_next(self, next_handler: "ImageHandler"):
        """Set the next handler on the chain"""
        self._next = next_handler
        return self


class CropHandler(ImageHandler):
    """Handles image cropping"""

    def _handle(self, image: np.ndarray, crop_bbox: tuple[point, point]) -> None:
        start, end = crop_bbox

        image = image[start[0] : start[1], end[0] : end[1]]


class ResizeHandler(ImageHandler):
    """Handles image resizing"""

    def _handle(self, image: np.ndarray, resize: dimension) -> None:
        image = cv.resize(image, resize, image, interpolation=cv.INTER_AREA)
