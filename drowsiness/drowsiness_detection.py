from detection.eye_insight_detector import EyeInsightDetector

class DrowsinessDetection:
    def __init__(self) -> None:
        # self._eye =
        # self._mouth =
        # self._head =
        ...

    def detectDrowsiness(self, images):
        eye = EyeInsightDetector(1, 0.15, 30)
        eye.execute(images)
