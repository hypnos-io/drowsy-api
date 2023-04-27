import time
from detection.head_detector import HeadDetector, create_frame_list
import cv2 as cv

head = HeadDetector()

start_time = time.time()

frame_sequence = create_frame_list("png")

current_time = time.time() - start_time
print(f"Tempo pra criar a lista de frames: {current_time} segundos\n")

start_time = time.time()

result_head = head.execute(frame_sequence)
print("=-=-=-=-=-=-=-=-=")
print("Detection Results: \n")
for key, value in result_head.data.items():
    print(f"{key}: {value:.2f}")
print("=-=-=-=-=-=-=-=-=")

current_time = time.time() - start_time
print(f"Tempo pra executar a detect: {current_time} segundos\n")

# classifier = KSSClassifier(len(frame_sequence), result_eye, result_head)
# response = classifier.classify(weights)
# print(response)
