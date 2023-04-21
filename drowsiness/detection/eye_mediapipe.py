import sys
sys.path.append(r'C:/Users/Callidus/Documents/drowsy-api')
import glob
from time import time
import itertools

import cv2 as cv
import numpy as np
import mediapipe as mp

from detector import MediapipeDetector

# def load_image(image):
#     return cv.imread(image, v.IMREAD_GRAYSCALE)


# def create_frame_list(location, extension):
#         images = glob.glob(f"detection/api/frames/{location}/*.{extension}")
        
#         frames = [cv.imread(image) for image in images]
        
#         frames = [cv.cvtColor(frame, cv.COLOR_BGR2RGB) for frame in frames]

#         return frames


class EyeDetector(MediapipeDetector):
    def __init__(self, blink_threshold, fps=10, ear_threshold=0.20):
        super().__init__()
        self._ear_treshhold = ear_threshold
        self._blink_threshold = blink_threshold
        
        self.frames = []
        self._frame_rate = fps
        self._frame_length = 1 / self._frame_rate
        
    def __calculate_left_ear__(self, left_eye):
        """Calcula o EAR (Eye Aspect Ratio) do olho esquerdo utilizando a fórmula:
        EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
           P1       P2
              _____  
             /     \ 
            /       \ 
        P4 \         / P3
            \       / 
             \_____/  
             P5    P6
"""
        # EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
        left_eye = np.array(left_eye)
        p2_minus_p6 = np.linalg.norm(left_eye[1] - left_eye[13])
        p3_minus_p5 = np.linalg.norm(left_eye[3] - left_eye[10])
        p1_minus_p4 = np.linalg.norm(left_eye[7] - left_eye[6])
        ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
        return ear

    def __calculate_right_ear__(self, right_eye):
        """Calcula o EAR (Eye Aspect Ratio) do olho direito utilizando a fórmula:
                EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
    P1      P2
       ____  
     /      \ 
   /          \ 
P4 \          / P3
    \        / 
      \____/  
      P5    P6

        """
        # EAR = (|P2-P6| + |P3-P5|) / (2 * |P1-P4|)
        right_eye = np.array(right_eye)
        p2_minus_p6 = np.linalg.norm(right_eye[0] -  right_eye[7])
        p3_minus_p5 = np.linalg.norm(right_eye[14] - right_eye[10])
        p1_minus_p4 = np.linalg.norm(right_eye[1] -  right_eye[4])
        
        ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
        return ear
    
    def __get_eyes__(self, results, frame):
        LEFT_EYE_INDEXES = list(set(itertools.chain(*self.mp_face_mesh.FACEMESH_LEFT_EYE)))
        RIGHT_EYE_INDEXES = list(set(itertools.chain(*self.mp_face_mesh.FACEMESH_RIGHT_EYE)))
        left_eye_landmarks = [results.multi_face_landmarks[0].landmark[i] for i in LEFT_EYE_INDEXES]
        right_eye_landmarks = [results.multi_face_landmarks[0].landmark[i] for i in RIGHT_EYE_INDEXES]

        # Converte landmarks para posição em pixels
        height, width, _ = frame.shape
        left_eye_positions = [(int(l.x * width), int(l.y * height)) for l in left_eye_landmarks]
        right_eye_positions = [(int(l.x * width), int(l.y * height)) for l in right_eye_landmarks]
        
        return left_eye_positions, right_eye_positions
    
    def _handle_frame(self, frame):
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = self.face_mesh_images.process(frame)
        EAR = 0.0
        data = {}
        
        if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
            
                    left_eye_pos, right_eye_pos = self.__get_eyes__(results, frame)
                    left_eye_ear = self.__calculate_left_ear__(left_eye_pos)
                    right_eye_ear = self.__calculate_right_ear__(right_eye_pos)
                    EAR = (left_eye_ear + right_eye_ear) / 2
                    EAR = round(EAR, 2)
                    data["ear"] = EAR
        return data
    
    def execute(self, images):
        detection_data = {"blink_count": 0, "closed_frame_count": 0}

        frame_data = []
        blink_frames = 0
        for frame in images:
            data = self._handle_frame(frame)

            if data["ear"] < self._ear_treshhold:
                blink_frames += 1
                detection_data["closed_frame_count"] += 1
            else:
                if blink_frames >= self._blink_threshold:
                    detection_data["blink_count"] += 1
                blink_frames = 0

            frame_data.append(data)

        detection_data["ear_mean"] = np.mean(data["ear"] for data in frame_data)
        detection_data["closed_time"] = (
            detection_data["closed_frame_count"] * self._frame_length
        )

        result = 0
        return DetectionData(result, detection_data)

if __name__ == "__main__":
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a camera")
        exit()

    detector = EyeDetector(1)
    prev = 0
    capture = True
    while capture:
        time_elapsed = time() - prev
        ret, frame = cap.read()

        if not ret:
            print("Não foi possivel capturar imagens da camera. Encerrando execução.")
            break

        key = cv.waitKey(1)

        if key == ord("q"):
            cap.release()
            capture = False

        if time_elapsed > 1.0 / detector._frame_rate:
            prev = time()

            data = detector._handle_frame(frame)

            y = 20
            for key, value in data.items():
                cv.putText(
                    frame,
                    f"{key}: {value:.2f}",
                    (10, y),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    1,
                    2,
                )
                y += 20

            cv.imshow("frame", frame)

    cv.destroyAllWindows()

# if __name__ == "__main__":

#     cap = cv.VideoCapture(0)

#     if not cap.isOpened():
#         print("Erro ao abrir a camera")
#         exit()

#     detector = EyeDetector(closed_eyes_threshold=2, blink_threshold=4)
#     prev = 0
#     capture = True
#     while capture:
#         time_elapsed = time() - prev
#         ret, frame = cap.read()

#         if not ret:
#             print("Não foi possivel capturar imagens da camera. Encerrando execução.")
#             break

#         key = cv.waitKey(1)

#         if key == ord("q"):
#             cap.release()
#             capture = False

#         if time_elapsed > 1.0 / detector.fps:
#             prev = time()

#             data = detector.execute(frame)

#             y = 20
#             for key, value in data.items():
#                 cv.putText(
#                     frame,
#                     f"{key}: {value:.2f}",
#                     (10, y),
#                     cv.FONT_HERSHEY_SIMPLEX,
#                     0.8,
#                     (255, 0, 0),
#                     1,
#                     2,
#                 )
#                 y += 20

#             cv.imshow("frame", frame)

#     cv.destroyAllWindows()