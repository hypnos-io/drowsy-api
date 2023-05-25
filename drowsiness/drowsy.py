from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

from ws.entities import FatigueStatus
from drowsiness.handlers import ResizeHandler, CropHandler
from drowsiness.classification import KSSClassifier
from drowsiness.detection import detector, eye, head, mouth


classifier = KSSClassifier(0, 0, 0)
handlers = CropHandler(ResizeHandler)

# Images per second per camera
FRAME_RATE = 10

# How many landmarks to process through the *.execute calls
BATCH_SIZE = FRAME_RATE * 10  # 30 times the amount of images in a second = 30 seconds
insight_face = detector.InsightDetector
mediapipe = detector.MediapipeDetector

cameras = {}


def reset_user(user_id):
    cameras[user_id] = {"count": 0, "insight": [], "mediapipe": []}


def add_landmarks(user_id, in_results, mp_results):
    cameras[user_id]["count"] += 1
    cameras[user_id]["insight"] += in_results
    cameras[user_id]["mediapipe"] += mp_results


def detect(
    user_id: int, video: list[np.ndarray], callback: Callable[[FatigueStatus], None], *args
) -> FatigueStatus:
    with ThreadPoolExecutor() as executor:
        in_faces = executor.map(insight_face["faces"], video)
        mp_landmarks = executor.map(mediapipe["images"].process, video)

        results = zip(in_faces, mp_landmarks)

        for in_face, mp_landmarks in results:
            in_landmarks = insight_face["landmarks"](in_face)

            add_landmarks(user_id, in_landmarks, mp_landmarks)

        if user_id in cameras and cameras[user_id]["count"] == BATCH_SIZE:
            eye_result = eye.execute(cameras[user_id]["insight"], fps=FRAME_RATE)
            mouth_result = mouth.execute(
                cameras[user_id]["insight"], fps=FRAME_RATE
            )
            head_result = head.execute(
                cameras[user_id]["mediapipe"], video[0].shape, fps=FRAME_RATE
            )

            reset_user(user_id)

            classifier.set_results(eye_result, head_result, mouth_result)
            callback(classifier.status(), *args)
        elif user_id not in cameras:
            reset_user(user_id)
