from eye_detector import EyeDetector, create_frame_list
import time

eye = EyeDetector(2, "detection/predictor/shape_predictor_68_face_landmarks.dat")
start_time = time.time()

frame_sequence = create_frame_list("warning")

current_time = time.time() - start_time
print(f'sequence: {current_time} seconds\n')

# frame = []
eye.execute(frame_sequence)
