import numpy as np

from detection.handlers import ResizeHandler, CropHandler
from detection.classification import KSSClassifier
from detection.detection.eye_detector import EyeDlibDetector
from detection.detection.mouth_detector import MouthDlibDetector
from detection.detection.head_detector  import HeadDetector

class Drowsy:
    def __init__(self, fps: int = 24) -> None:
        self.fps = fps

        self.detectors = (
            EyeDlibDetector(1, fps),
            MouthDlibDetector(fps),
            HeadDetector()
        )
        
        self.classifier = KSSClassifier()
        self.handler = CropHandler(ResizeHandler)

    def _run_detectors(self, image):
        for detector in self.detectors:
            detector.



    def detect(self, video: list[np.ndarray]):
        
        for frame in video:
             self._run_detectors