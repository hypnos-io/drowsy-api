from image_handler import ImageHandler
from drowsy_types import CV2Image
class CropHandler(ImageHandler):
    """Handles image cropping"""

    def _handle(self, image: CV2Image) -> None:
        raise NotImplementedError