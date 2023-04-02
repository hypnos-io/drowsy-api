from eye_detector import EyeDetector, create_frame_list


eye = EyeDetector(2, "detection/predictor/shape_predictor_68_face_landmarks.dat")


frame_sequence = create_frame_list()

time = eye.closed_eyes(2, frame_sequence)

print(time)