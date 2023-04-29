import numpy as np

from detection.detection import detector, eye_mediapipe

# [0]P1 [1]P2 [2]P3 [3]P4 [4]P5 [5]P6
left_points = np.array([35, 41, 42, 39, 37, 36])
right_points = np.array([89, 95, 96, 93, 91, 90])

class EyeInsightDetector(detector.InsightDetector):
    def __init__(self, ear_threshold=0.17, closed_eyes_threshold=3, fps=10, video_lenght=30):
        super().__init__()
        
        self._blink_max = 20
        self.__frame_length = 1 / fps
        self.__ear_threshold = ear_threshold
        #self.__blink_threshold = blink_threshold
        self.__video_length = video_lenght
        self.__closed_eyes_threshold = closed_eyes_threshold
        
    def __calculate_ear__(self, eye):
        # eye: [0]P1 [1]P2 [2]P3 [3]P4 [4]P5 [5]P6
        """EAR = (|P2 - P6| + |P3 - P5|) / (2 * |P1 - P4|)"""
        vertical_distl = np.linalg.norm(eye[1] - eye[-1])
        vertical_distr = np.linalg.norm(eye[2] - eye[-2])
        horizontal_dist = np.linalg.norm(eye[0] - eye[3])
        eye_aspect_ratio = (vertical_distl + vertical_distr) / (2.0 * horizontal_dist)

        return eye_aspect_ratio
    
    def _handle_frame_(self, frame):
        faces = self._detect_faces(frame)
        #tim = frame.copy()
        average_ear = None

        for face in faces:
                landmarks = self._detect_landmarks(face)
                left_eye = np.array(landmarks[left_points])
                right_eye = np.array(landmarks[right_points])
                
                left_ear = self.__calculate_ear__(left_eye)
                right_ear = self.__calculate_ear__(right_eye)
                average_ear = np.mean((left_ear, right_ear))
                
        return average_ear
            
    def execute(self, images) -> detector.DetectionData:
        detection_data = {"eye_opening": 0.0, "blink_count": 0, "closed_eyes_time": 0.0}
        weights = {"eye_opening_weight": 0.4,"blink_count_weight": 0.2, "close_eyes_time_weight": 0.4}
        ear_list = []
        close_frames = 0
        i = 0
        for frame in images:
            i += 1
            ear = self._handle_frame_(frame)
            if ear is not None:
                ear_list.append(ear)
                if ear < self.__ear_threshold:
                    close_frames += 1
                    #print(f"closed at frame_{i} | sequence: {close_frames}")
                    detection_data["closed_eyes_time"] += 1
                else:
                    if 1 <= close_frames < self.__closed_eyes_threshold:
                        detection_data["blink_count"] += 1
                    close_frames = 0
        
        if not ear_list:
            return detector.DetectionData(0, {"blink_count": 0, "eye_opening": 0, "closed_eyes_time": 0})

        ear_array = np.array(ear_list)
        ear_min = np.min(ear_array)
        ear_max = np.max(ear_array)
        ear_norm = abs(ear_array - ear_max) / abs(ear_min - ear_max)
        average_ear = np.mean(ear_norm)

        detection_data["eye_opening"] = average_ear
        detection_data["closed_eyes_time"] = (detection_data["closed_eyes_time"] * self.__frame_length) / self.__video_length
        detection_data["blink_count"] /= self._blink_max
        

        result = (detection_data["eye_opening"] * weights["eye_opening_weight"]
                + detection_data["closed_eyes_time"] * weights["close_eyes_time_weight"]
                + detection_data["blink_count"] * weights["blink_count_weight"])
        
        return detector.DetectionData(round(result, 1), detection_data)
            
if __name__ == '__main__':
    eye_detector = EyeInsightDetector(ear_threshold=0.17, closed_eyes_threshold=3, fps=10, video_lenght=32)
    
    frame_sequence = eye_mediapipe.create_frame_list("test/not_tired", "png")
    if len(frame_sequence) <= 0:
        print("Lista vazia.")
    else:
        print("\nRunning detection....")
        response = eye_detector.execute(frame_sequence)
        print("=" * 30 + " RESULTS: " + 30 * "=")    
        response.show()
        if response.result < 0.4:
            print("Not tired")
        elif 0.4 <= response.result < 0.7:
            print("Kinda tired")
        else:
            print("Tired")
