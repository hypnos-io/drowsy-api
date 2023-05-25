from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np

from ws.entities import FatigueStatus
from drowsiness.handlers import ResizeHandler, CropHandler
from drowsiness.classification import KSSClassifier
from drowsiness.detection import detector, eye, head, mouth


classifier = KSSClassifier(0, 0, 0)
handlers = CropHandler(ResizeHandler)

FRAME_RATE = 10
insight_face = detector.InsightDetector
mediapipe = detector.MediapipeDetector


def detect(video: list[np.ndarray]) -> FatigueStatus:
    print("detecting")
    mp_results = None
    with ThreadPoolExecutor() as executor:
        faces = executor.map(insight_face["faces"], video)
        mp_results = executor.map(mediapipe["images"].process, video)

    in_results = map(insight_face["landmarks"], faces)

    eye_result = eye.execute(in_results, fps=FRAME_RATE)
    mouth_result = mouth.execute(in_results, fps=FRAME_RATE)
    head_result = head.execute(mp_results, video[0].shape, fps=FRAME_RATE)

    classifier.set_results(eye_result, head_result, mouth_result)

    return classifier.status()
