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
    def execute(self, source):
        pass

