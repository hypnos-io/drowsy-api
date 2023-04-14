import cv2 as cv

from drowsiness.drowsy_types import CV2Image

from image_handler import ImageHandler


class ColorHandler(ImageHandler):
    """Handles image colorspace conversion"""

    def _handle(self, image: CV2Image) -> None:
        cv.cvtColor(image, cv.COLOR_RGB2GRAY, image)
