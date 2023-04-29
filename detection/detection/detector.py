import json
from os import path
from abc import ABC, abstractmethod

import numpy as np
import mediapipe as mp
import dlib
import insightface
from insightface.app import FaceAnalysis


MODULE_DIR = path.dirname("c:/Users/Callidus/Documents/drowsy-api/drowsiness/predictor")
path.dirname(path.abspath(__file__))
PREDICTOR_FACE_68 = dlib.shape_predictor(
    path.join(MODULE_DIR, "predictor", "shape_predictor_68_face_landmarks.dat")
)
DETECTOR_FHOG = dlib.get_frontal_face_detector()


class AbstractDetector(ABC):
    @abstractmethod
    def execute(self, images):
        pass


class InsightDetector(AbstractDetector):
    def __init__(self):
        self.__app = FaceAnalysis(
            allowed_modules=["detection", "landmark_2d_106"],
            providers=["CPUExecutionProvider"],
        )
        self.__app.prepare(ctx_id=0, det_size=(640, 640))

    def _detect_faces(self, source):
        faces = self.__app.get(source)
        return faces

    def _handle_frame(self, frame) -> float:
        pass

    def _detect_landmarks(self, face):
        landmarks = face.landmark_2d_106
        landmarks = np.round(landmarks).astype(int)

        return landmarks


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


class MediapipeEyeDetector(AbstractDetector):
    def __init__(self) -> None:
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh_images = self.mp_face_mesh.FaceMesh()


class MediapipeHeadDetector(AbstractDetector):
    def __init__(self) -> None:
        self.mp_pose = mp.solutions.pose
        self.pose_images = self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )


class DetectionData:
    def __init__(self, result, data) -> None:
        assert 0 <= result <= 1, "Detection result should be inbetween 0 and 1"

        self.result = result
        self.data = data

    def json(self) -> str:
        return json.dumps({"result": self.result, **self.data})
