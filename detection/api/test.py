from eye_mediapipe import EyeDetector, create_frame_list
import time

# eye = EyeDetector(2, "detection/predictor/shape_predictor_68_face_landmarks.dat")
# start_time = time.time()

# frame_sequence = create_frame_list("warning")

# current_time = time.time() - start_time
# print(f'sequence: {current_time} seconds\n')

# # frame = []
# eye.execute(frame_sequence)

eye = EyeDetector(3, 1)

start_time = time.time()

frame_sequence = create_frame_list("warning", "png")

current_time = time.time() - start_time
print(f'Tempo pra criar a lista de frames: {current_time} segundos\n')

start_time = time.time()

result = eye.execute(frame_sequence)

current_time = time.time() - start_time
print(f'Tempo pra executar a detecção: {current_time} segundos\n')

print(result)
