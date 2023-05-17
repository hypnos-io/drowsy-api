import numpy as np

#from ws.entities import FatigueStatus

from detection.handlers import ResizeHandler, CropHandler
from detection.classification import KSSClassifier
from detection.detection.eye_insight_detector import EyeInsightDetector
from detection.detection.mouth_insight_detector import MouthInsightDetector
from detection.detection.head_detector import HeadDetector


class Drowsy:
    def __init__(self, fps: int = 24) -> None:
        self.fps = fps

        self._eye = EyeInsightDetector(fps=fps)
        self._mouth = MouthInsightDetector(fps=fps)
        self._head = HeadDetector(fps=fps)

        self.classifier = KSSClassifier(0, 0, 0)
        self.handler = CropHandler(ResizeHandler)

    def detect(self, video: list[np.ndarray]) -> FatigueStatus:
        eye_result = self._eye.execute(video)
        mouth_result = self._mouth.execute(video)
        head_result = self._head.execute(video)

        self.classifier.set_results(
            eye_result,
            head_result,
            mouth_result
        )

        kss = self.classifier.classify()

        return {
            "kssScale": kss,
            "detection": {
                "eyes": eye_result.to_dict(),
                "head": head_result.to_dict(),
                "mouth": mouth_result.to_dict()
            }
        }
        
if __name__ == "__main__":
    print("Hi")