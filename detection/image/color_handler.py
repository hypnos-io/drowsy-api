from detection.image.image_handler import CV2Image
from image_handler import ImageHandler
import cv2

class ColorHandler(ImageHandler):
    """Handles image colorspace conversion"""

    def _handle(self, image: CV2Image) -> None:
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY, image)