from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np

from ws.entities import FatigueStatus
from drowsiness.handlers import ResizeHandler, CropHandler
from drowsiness.classification import KSSClassifier
from drowsiness.detection import detector, eye, head, mouth


classifier = KSSClassifier(0, 0, 0)
handlers = CropHandler(ResizeHandler)

FRAME_RATE = 24
insight_face = detector.InsightDetector
mediapipe = detector.MediapipeDetector

def detect(video: list[np.ndarray]) -> FatigueStatus:
    mp_results = None
    with ThreadPoolExecutor() as executor:
        faces = executor.map(insight_face['faces'], video)
        mp_results = executor.map(mediapipe['images'].process, video)

    in_results = map(insight_face['landmarks'], faces)

    eye_result = eye.execute(in_results)
    mouth_result = mouth.execute(in_results)
    head_result = head.execute(mp_results, video[0].shape)

    classifier.set_results(
        eye_result,
        head_result,                   
        mouth_result      
    )

    return classifier.status()

    
