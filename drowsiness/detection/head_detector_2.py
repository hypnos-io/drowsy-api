import sys
sys.path.append(r'C:/Users/Callidus/Documents/drowsy-api')
from detector import DetectionData
import glob
from time import time
import itertools

import cv2 as cv
import numpy as np
import mediapipe as mp

from detector import MediapipeHeadDetector

class HeadDetector(MediapipeHeadDetector):
    def __init__(self, fps=60, angle_threshold=110, inclination_side_threshold=40):
        super().__init__()
        self.angle_threshold = angle_threshold
        self.inclination_threshold = inclination_side_threshold
        
        self.frames = []
        self._frame_rate = fps
        self._frame_length = 1 / self._frame_rate

    def __get_head__(self, results, frame):
        RIGHT_EAR_INDEX = self.mp_pose.PoseLandmark.RIGHT_EAR
        LEFT_EAR_INDEX = self.mp_pose.PoseLandmark.LEFT_EAR
        NOSE_INDEX = self.mp_pose.PoseLandmark.NOSE

        right_ear_landmarks = results.pose_landmarks.landmark[RIGHT_EAR_INDEX] 
        left_ear_landmarks = results.pose_landmarks.landmark[LEFT_EAR_INDEX] 
        nose_landmarks = results.pose_landmarks.landmark[NOSE_INDEX]  

        height, width, _ = frame.shape
        right_ear_positions = (int(right_ear_landmarks.x * width), int(right_ear_landmarks.y * height))
        left_ear_positions = (int(left_ear_landmarks.x * width), int(left_ear_landmarks.y * height))
        nose_positions = (int(nose_landmarks.x * width), int(nose_landmarks.y * height))

        return right_ear_positions, left_ear_positions, nose_positions
    
    def __calculate_head_angle__(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
            a[1] - b[1], a[0] - b[0]
        )
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def __calculate_head_inclination__(self, a, b):
        a = np.array(a)
        b = np.array(b)

        radians = np.arctan2(b[1] - a[1], b[0] - a[0])
        angle = np.degrees(radians)

        if angle < 0:
            angle = angle * -1

        return angle
    
    def _handle_frame(self, frame):
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results = self.pose_images.process(frame)
        frame.flags.writeable = True
        data = {
            "head_angle": 0,
            "head_inclination": 0
        }
        
        if results.pose_landmarks:
                for pose_landmarks in results.pose_landmarks.landmark:
                    right_ear_pos, left_ear_pos, nose_pos = self.__get_head__(results, frame)
                    head_angle = self.__calculate_head_angle__(right_ear_pos, nose_pos, left_ear_pos)
                    head_inclination = self.__calculate_head_inclination__(right_ear_pos, left_ear_pos)
                    data["head_angle"] = head_angle
                    data["head_inclination"] = head_inclination
        return data
    
    def execute(
            self, 
            images,
            consec_frames_threshold_angle=20,
            consec_side_threshold_angle=2):
        
        detection_data = {
            "total_angle_down_time": 0,
            "total_inclination_down_time": 0,
            "head_angle_mean": 0,
            "head_inclination_mean": 0,
            "angle_down_consecutives": 0,
            "inclination_down_consecutives": 0,
            "angle_down_count": 0,
            "inclination_down_count": 0,
            }
        
        frame_data = []
        for frame in images:
            data = self._handle_frame(frame)
            
            # Head Angle
            if data["head_angle"] < self.angle_threshold:
                detection_data["angle_down_consecutives"] += 1

                if detection_data["angle_down_consecutives"] > consec_frames_threshold_angle:
                            detection_data["angle_down_count"] += 1

            else:
                detection_data["angle_down_consecutives"] = 0
                if detection_data["angle_down_count"] > 1:
                        detection_data["angle_down_count"] = 0


            # # Head Inclination
            if data["head_inclination"] < self.inclination_threshold:
                detection_data["inclination_down_consecutives"] += 1

                if detection_data["inclination_down_consecutives"] == consec_side_threshold_angle:
                    detection_data["inclination_down_count"] += 1 
            
            else:
                detection_data["inclination_down_consecutives"] = 0
                if detection_data["inclination_down_count"] > 1:
                    detection_data["inclination_down_count"] = 0

            frame_data.append(data)

        detection_data["head_angle_mean"] = np.mean([data["head_angle"] for data in frame_data])
        detection_data["total_angle_down_time"] = (
            detection_data["angle_down_count"] * self._frame_length
        )
        detection_data["head_inclination_mean"] = np.mean([data["head_inclination"] for data in frame_data])
        detection_data["total_inclination_down_time"] = (
            detection_data["total_inclination_down_time"] * self._frame_length
        )
        result = 0
    

        return DetectionData(result, detection_data)

if __name__ == "__main__":

    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a camera")
        exit()

    detector = HeadDetector()
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

            data = detector.execute(frame)

            y = 20
            for key, value in data.data.items():
                cv.putText(
                    frame,
                    f"{key}: {value:.2f}",
                    (10, y),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
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

#     detector = HeadDetector()
#     prev = 0
#     capture = True

#     detection_data = {
#             "total_angle_down_time": 0,
#             # "total_inclination_down_time": 0,
#             "head_angle_mean": 0,
#             # "head_inclination_mean": 0,
#             "angle_down_consecutives": 0,
#             # "inclination_down_consecutives": 0,
#             "angle_down_count": 0,
#             # "inclination_down_count": 0,
#             }
    
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

#         if time_elapsed > 1.0 / detector._frame_rate:
#             prev = time()

#             data = detector._handle_frame(frame)

#             y = 20
#             for key, value in data.items():
#                 cv.putText(
#                     frame,
#                     f"{key}: {value:.2f}",
#                     (10, y),
#                     cv.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     (255, 0, 0),
#                     1,
#                     2,
#                 )
#                 y += 20

#             cv.imshow("frame", frame)

#     cv.destroyAllWindows()
