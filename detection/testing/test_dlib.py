# from api.eye_mediapipe import EyeDetector, create_frame_list
import time

from detection.detection.eye_detector import EyeDetector, create_frame_list

# test_image = cv2.imread("detection/api/frames/tests/half_eyes_2.png")
# cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

weights = {
    "closed eye weight": 0.4,
    "blink weight": 0.2,
    "eye ear weight": 0.1,
    "head tilt weight": 0.4,
}

eye = EyeDetector(
    closed_eyes_threshold=5,
    blink_threshold=2,
    landmarks_model_path="detection/predictor",
)

start_time = time.time()

frame_sequence = create_frame_list("tests/closed", "png")
print(len(frame_sequence))
# frame_sequence = [test_image]

current_time = time.time() - start_time
print(f"Tempo pra criar a lista de frames: {current_time} segundos\n")

start_time = time.time()

result_eye = eye.execute(frame_sequence)
print(result_eye)

# result_head = head.execute(frame_sequence)
# print(result_head)

current_time = time.time() - start_time
print(f"Tempo pra executar a detect: {current_time} segundos\n")

# classifier = KSSClassifier(len(frame_sequence), result_eye, result_head)
# response = classifier.classify(weights)
# print(response)
