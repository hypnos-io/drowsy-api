"""
Image Handlers
"""

from abc import ABC, abstractmethod
from typing import Optional

from drowsy_types import CV2Image


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

    def set_next(self, next_handler: 'ImageHandler'):
        """Set the next handler on the chain"""
        self._next = next_handler
        return self