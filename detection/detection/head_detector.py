import glob
from time import time

import cv2 as cv
import numpy as np

from detection.detection import detector

# def load_image(image):
#     return cv.imread(image, v.IMREAD_GRAYSCALE)

def create_frame_list(extension):
        images = glob.glob(f"C:/Users/callidus/drowsy-api/drowsiness/detection/test/frames/*.{extension}")
        
        frames = [cv.imread(image) for image in images]
        
        frames = [cv.cvtColor(frame, cv.COLOR_BGR2RGB) for frame in frames]

        return frames

class HeadDetector(detector.MediapipeHeadDetector):
    def __init__(self, fps=10, frontal_threshold=110, lateral_threshold=35):
        super().__init__()
        self.frontal_threshold = frontal_threshold
        self.lateral_threshold = lateral_threshold
        
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
    
    def __calculate_head_frontal__(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
            a[1] - b[1], a[0] - b[0]
        )
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle
        
        if b[1] < a[1] - 17 or b[1] < c[1] - 17:
            angle = 140

        return angle

    def __calculate_head_lateral__(self, a, b):
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
            "head_frontal": 0,
            "head_lateral": 0
        }
        
        if results.pose_landmarks:
                for pose_landmarks in results.pose_landmarks.landmark:
                    right_ear_pos, left_ear_pos, nose_pos = self.__get_head__(results, frame)
                    head_frontal = self.__calculate_head_frontal__(right_ear_pos, nose_pos, left_ear_pos)
                    head_lateral = self.__calculate_head_lateral__(right_ear_pos, left_ear_pos)
                    data["head_frontal"] = head_frontal
                    data["head_lateral"] = head_lateral
        return data
    
    def execute(
            self, 
            images,
            consec_frames_threshold_frontal=2,
            consec_frames_threshold_lateral=2):
        
        detection_data = {
            "total_frontal_down_time": 0,
            "head_frontal_mean": 0,
            "total_lateral_down_time": 0,
            "head_lateral_mean": 0,
            }
        
        lateral_down_count = 0
        frontal_down_count = 0

        frontal_down_consecutives = 0
        lateral_down_consecutives = 0
        
        frame_data = []
        frames = 0
        for frame in images:
            frames += 1
            data = self._handle_frame(frame)
            
            # Head frontal
            if data["head_frontal"] < self.frontal_threshold:
                frontal_down_consecutives += 1

                if frontal_down_consecutives > consec_frames_threshold_frontal:
                            print(f"\033[31m Frontal :{frames}\033[0m")
                            frontal_down_count += 1

            else:
                frontal_down_consecutives = 0


            # # Head lateral
            if data["head_lateral"] > self.lateral_threshold:
                lateral_down_consecutives += 1

                if lateral_down_consecutives > consec_frames_threshold_lateral:
                    print(f"\033[32m Latereal :{frames}\033[0m")
                    lateral_down_count += 1 
            
            else:
                lateral_down_consecutives = 0

            frame_data.append(data)

        detection_data["head_frontal_mean"] = np.mean([data_frontal["head_frontal"] for data_frontal in frame_data])
        detection_data["total_frontal_down_time"] = (
            frontal_down_count * self._frame_length
        )
        detection_data["head_lateral_mean"] = np.mean([data_lateral["head_lateral"] for data_lateral in frame_data])
        detection_data["total_lateral_down_time"] = (
            lateral_down_count * self._frame_length
        )
        
        # Result
        frontal_max = np.max([data_frontal["head_frontal"] for data_frontal in frame_data])
        frontal = np.array(
             [((data_frontal["head_frontal"] - 0) / (frontal_max - 0)) for data_frontal in frame_data]
        )

        lateral_max = np.max([data_lateral["head_lateral"] for data_lateral in frame_data])
        lateral = np.array(
            [((data_lateral["head_lateral"] - 0) / (lateral_max - 0)) for data_lateral in frame_data]
        )

        final_result_array = np.array(
            [(frontal[i] * 0.6) + (lateral[i] * 0.4) for i in range(len(frame_data))]
        )

        for i in final_result_array:
            if i >= 10:
                print("Erro")

        final_result = np.mean(final_result_array)

        return detector.DetectionData(final_result, detection_data)

if __name__ == "__main__":

    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a camera")
        exit()

    head_detector = HeadDetector()
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

        if time_elapsed > 1.0 / head_detector._frame_rate:
            prev = time()

            data = head_detector._handle_frame(frame)

            y = 20
            for key, value in data.items():
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
