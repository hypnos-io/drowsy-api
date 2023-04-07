from eye_detector import EyeDetector, create_frame_list
from head_detector import HeadDetector
import time

eye = EyeDetector(2, "detection/predictor/shape_predictor_68_face_landmarks.dat")
head = HeadDetector(0.20)
start_time = time.time()

frame_sequence = create_frame_list("warning")

current_time = time.time() - start_time
print(f'sequence: {current_time} seconds\n')

# frame = []
result = head.execute(frame_sequence)
print(result)