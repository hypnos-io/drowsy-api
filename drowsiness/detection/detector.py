import numpy as np
import mediapipe as mp
import insightface
from insightface.app import FaceAnalysis


def insight_app():
    app = FaceAnalysis(
        allowed_modules=["detection", "landmark_2d_106", 'landmark_3d_68'],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def get2dmarks(insight_faces):
    if insight_faces:
        return np.round(insight_faces[0].landmark_2d_106).astype(int)

def get3dmarks(insight_faces):
    if insight_faces:
        return np.round(insight_faces[0].landmark_3d_68).astype(int)

INSIGHT_FACE = insight_app()

InsightDetector = {
    "faces": INSIGHT_FACE.get,
    "2d": get2dmarks,
    "3d": get3dmarks,
}

MediapipeDetector = {
    "pose": mp.solutions.pose,
}
MediapipeDetector["images"] = MediapipeDetector["pose"].Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)


class DetectionData:
    def __init__(self, result, data) -> None:
        #assert 0 <= result <= 1, "Detection result should be inbetween 0 and 1"

        self.result = result
        self.data = data

    def to_dict(self):
        return {"result": self.result, **self.data}


MediapipeDetector = {
    "pose": mp.solutions.pose,
}

MediapipeDetector["images"] = MediapipeDetector["pose"].Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)


# def insight_app():
#     app = FaceAnalysis(
#         allowed_modules=["detection", "landmark_2d_106"],
#         providers=["CPUExecutionProvider"],
#     )
#     app.prepare(ctx_id=0, det_size=(640, 640))
#     return app


# INSIGHT_FACE = insight_app()

# InsightDetector = {
#     "faces": INSIGHT_FACE.get,
#     "landmarks": lambda face: np.round(face.landmark_2d_106).astype(int),
# }
