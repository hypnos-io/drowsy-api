from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np

from ws.entities import FatigueStatus
from detection.handlers import ResizeHandler, CropHandler
from detection.classification import KSSClassifier
from detection.detection import detector, eye_insight_detector, mouth_detector, head_detector


classifier = KSSClassifier(0, 0, 0)
handlers = CropHandler(ResizeHandler)

FRAME_RATE = 24
insight_face = detector.InsightDetector

def detect(video: list[np.ndarray]) -> FatigueStatus:
    with ThreadPoolExecutor() as executor:
        faces = executor.map(insight_face['faces'], video)

    landmarks = map(insight_face['landmarks'], faces)

    eye_result = eye_insight_detector.execute(landmarks)
    # mouth_result = mouth_detector.execute(video)
    # head_result = head_detector.execute(video)

    classifier.set_results(
        eye_result,
        #head_result,                   
        #mouth_result      
    )

    return classifier.status()

    
