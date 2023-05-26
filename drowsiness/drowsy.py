from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from functools import partial
from itertools import chain
from typing import Callable

import numpy as np

from ws.entities import FatigueStatus
from drowsiness.handlers import ResizeHandler, CropHandler
from drowsiness.classification import KSSClassifier
from drowsiness.detection import detector, eye, head, mouth


classifier = KSSClassifier()
handlers = CropHandler(ResizeHandler)

# Images per second per camera
FRAME_RATE = 10

# How many landmarks to process through the *.execute calls
BATCH_SIZE = 3 # 30 times the amount of images in a second = 30 seconds
insight_face = detector.InsightDetector
mediapipe = detector.MediapipeDetector

cameras = {}


def reset_user(user_id):
    cameras[user_id] = {"count": 0, "insight2d": [], "insight3d": [], "add": partial(add_landmarks, user_id)}


def add_landmarks(user_id, results):
    in_2dmarks, in_3dmarks = results
    cameras[user_id]["insight2d"].append(in_2dmarks)
    cameras[user_id]["insight3d"].append(in_3dmarks)


def detect(
    user_id: str, video: list[np.ndarray], callback: Callable[[FatigueStatus], None], *args
) -> FatigueStatus:
    with ThreadPoolExecutor() as executor:
        in_faces = executor.map(insight_face["faces"], video)
        
        in_2dmarks = map(insight_face["2d"], in_faces)
        in_3dmarks = map(insight_face["3d"], in_faces)
        results = zip(in_2dmarks, in_3dmarks)
        
        if user_id not in cameras:
            reset_user(user_id)
        
        cameras[user_id]["count"] += 1
        if "results" in cameras[user_id]:
            cameras[user_id]["results"] = chain(cameras[user_id]["results"], map(cameras[user_id]["add"], results))
        else:
            cameras[user_id]["results"] = map(cameras[user_id]["add"], results)



        if cameras[user_id]["count"] == BATCH_SIZE:
            any(cameras[user_id]['results'])
            if (ins := cameras[user_id]['insight2d']) and len(ins) > 0:
                print('2D DETECTION')
                mouth_result = mouth.execute(
                    cameras[user_id]["insight2d"], fps=FRAME_RATE
                )
                eye_result = eye.execute(cameras[user_id]["insight2d"], fps=FRAME_RATE)
                classifier.set_results(eye=eye_result, mouth=mouth_result)

            if (mp := cameras[user_id]['insight3d']) and len(mp) > 0:
                print('3D DETECTION')
                head_result = head.execute(
                    cameras[user_id]["insight3d"], fps=FRAME_RATE
                )
                classifier.set_results(head=head_result)

            callback(classifier.status(), *args)

            reset_user(user_id)
            classifier.set_results(None, None, None)            
        elif cameras[user_id]["count"] > BATCH_SIZE: # Nunca deveria ocorrer
            reset_user(user_id)
