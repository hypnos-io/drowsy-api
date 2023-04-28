import time

import cv2 as cv

from detection.detection.eye_mediapipe import EyeDetector, create_frame_list
from detection.detection.head_detector import HeadDetector
from detection.classification import KSSClassifier

# test_image = cv.imread("detection/api/frames/tests/half_eyes_2.png")
# cv.cvtColor(test_image, cv.COLOR_BGR2RGB)
# weights = {
#     "closed eye weight": 0.6,
#     "blink weight": 0.2,
#     "eye ear weight": 0.2,
#     "head_tilt_weight": 0.4
# }

eye = EyeDetector(closed_eyes_threshold=5, blink_threshold=2)
head = HeadDetector(0.3)

start_time = time.time()

frame_sequence = create_frame_list("tests/closed","png")
print(len(frame_sequence))
# # frame_sequence = [test_image]
current_time = time.time() - start_time
print(f'Tempo pra criar a lista de frames: {current_time} segundos\n')

start_time = time.time()

result_eye = eye.execute(frame_sequence)
print(result_eye)

# result_head = head.execute(frame_sequence)
# print(result_head)

current_time = time.time() - start_time
print(f'Tempo pra executar a detect: {current_time} segundos\n')

# classifier = KSSClassifier(300, detection_dict, result_dict)
# response = classifier.classify(weights)
# print(response)
