import json
from os import path
from abc import ABC, abstractmethod

import dlib


MODULE_DIR = path.dirname(path.abspath(__file__))
PREDICTOR_FACE_68 = dlib.shape_predictor(
    path.join(MODULE_DIR, "predictor", "shape_predictor_68_face_landmarks.dat")
)
DETECTOR_FHOG = dlib.get_frontal_face_detector()


class AbstractDetector(ABC):
    @abstractmethod
    def execute(self, images):
        pass

class DlibDetector(AbstractDetector):
    def __init__(self) -> None:
        self._detector = DETECTOR_FHOG
        self._predictor = PREDICTOR_FACE_68

    def _detect_faces(self, source):
        faces = self._detector(source)

        return faces

    def _detect_landmarks(self, source, face):
        landmarks = self._predictor(source, face)

        return landmarks

    def point_tuple(point):
        """Auxiliary method to convert Dlib Point object into coordinate tuple used by OpenCV"""
        return (point.x, point.y)


class DetectionData:
    def __init__(self, result, data) -> None:
        assert 0 <= result <= 1, "Detection result should be inbetween 0 and 1"

        self.result = result
        self.data = data

    def json(self) -> str:
        return json.dumps({"result": self.result, **self.data})