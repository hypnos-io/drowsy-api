from os import path
from abc import ABC, abstractmethod

import numpy as np
import mediapipe as mp
import insightface
from insightface.app import FaceAnalysis

class AbstractDetector(ABC):
    @abstractmethod
    def execute(self, images):
        pass

INSIGHT_FACE = FaceAnalysis(
    allowed_modules=["detection", "landmark_2d_106"],
    providers=["CPUExecutionProvider"],
)
INSIGHT_FACE.prepare(ctx_id=0, det_size=(640, 640))

InsightDetector = {
    'faces': INSIGHT_FACE.get,
    'landmarks': lambda face: np.round(face.landmark_2d_106).astype(int)
}

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

    def to_dict(self):
        return {"result": self.result, **self.data}
