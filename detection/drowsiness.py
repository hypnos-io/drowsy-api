from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np

from ws.entities import FatigueStatus
from detection.handlers import ResizeHandler, CropHandler
from detection.classification import KSSClassifier
from detection.detection import detector, eye, head, mouth


classifier = KSSClassifier(0, 0, 0)
handlers = CropHandler(ResizeHandler)

FRAME_RATE = 24
insight_face = detector.InsightDetector

def detect(video: list[np.ndarray]) -> FatigueStatus:
    with ThreadPoolExecutor() as executor:
        faces = executor.map(insight_face['faces'], video)

    landmarks = map(insight_face['landmarks'], faces)

    eye_result = eye.execute(landmarks)
    mouth_result = mouth.execute(landmarks)
    head_result = head.execute(video)

    classifier.set_results(
        eye_result,
        head_result,                   
        mouth_result      
    )

    return classifier.status()

    
