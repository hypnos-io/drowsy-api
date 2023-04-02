from image_handler import ImageHandler

class CropHandler(ImageHandler):
    """Handles image cropping"""

    def _handle(self, image: CV2Image) -> None:
        raise NotImplementedError