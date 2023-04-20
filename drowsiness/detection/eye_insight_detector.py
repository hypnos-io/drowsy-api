import cv2
import numpy as np
import sys
sys.path.append(r'C:/Users/Callidus/Documents/drowsy-api')

from detector import InsightDetector
from drowsiness.classification.detection_data import DetectionData

# [0]P1 [1]P2 [2]P3 [3]P4 [4]P5 [5]P6
left_points = [35, 41, 42, 39, 37, 36]
right_points = [89, 95, 96, 93, 91, 90]

class EyeInsightDetector(InsightDetector):
    def __init__(self, blink_threshold, ear_threshold, fps):
        super().__init__()
        
        self.__frame_rate = fps
        self.__frame_length = 1 / fps
        self.__ear_threshold = ear_threshold
        self.__blink_threshold = blink_threshold
        self.__detection_data = {
            "eye_opening": [0.0],
            "blink_count": 0.0,
            "closed_eyes_time": 0.0
        }
        
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
        tim = frame.copy()
        ear = []

        for face in faces:
                landmarks = self._detect_landmarks(face)
                for i in range(landmarks.shape[0]):
                    p = tuple(landmarks[i])
                    cv2.circle(frame, p, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    
                left_eye = np.array(landmarks[left_points])
                right_eye = np.array(landmarks[right_points])
                
                left_ear = self.__calculate_ear__(left_eye)
                right_ear = self.__calculate_ear__(right_eye)
                ear =  np.mean((left_ear, right_ear)) 
        cv2.putText(frame, f'eye aspect ratio: {ear}', (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
        return ear
    
    def execute(self, images):
        detection_data = {"blink_count": 0, "closed_frame_count": 0}
        frame_data = []
        close_frames = 0
        
        for frame in images:
            ear = self._handle_frame_(frame)
            
            if ear != None:
                if ear < self.__ear_threshold:
                    close_frames += 1
                    print(f"closed: {ear:.2f}")
                    detection_data["closed_frame_count"] += 1
                else:
                    if close_frames >= self.__blink_threshold:
                        detection_data["blink_count"] += 1
                    close_frames = 0
                frame_data.append(ear)
            
        self.__detection_data["eye_opening"] = np.mean(ear for data in frame_data)
        self.__detection_data["closed_eyes_time"] = (
            detection_data["closed_frame_count"] * self.__frame_length
        )
        
        return self.__detection_data
    
    def test(self):
            detection_data = {"blink_count": 0, "closed_frame_count": 0}
            frame_data = []
            close_frames = 0
            
            cap = cv2.VideoCapture(0) 

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                ear = self._handle_frame_(frame)
                if ear != None:
                    if ear < self.__ear_threshold:
                        close_frames += 1
                        detection_data["closed_frame_count"] += 1
                    else:
                        if close_frames >= self.__blink_threshold:
                            detection_data["blink_count"] += 1
                        close_frames = 0
                cv2.imshow('InsightFace', frame)

                if cv2.waitKey(1) == ord('q'):
                    break
                
                frame_data.append(ear)
            
            self.__detection_data["blink_count"] = detection_data["blink_count"]
            self.__detection_data["closed_eyes_time"] = (
                detection_data["closed_frame_count"] * self.__frame_length
            )
            
            cap.release()
            cv2.destroyAllWindows()
            
            print(self.__detection_data)
            
            
if __name__ == '__main__':
    detector = EyeInsightDetector(1, 0.14, 30)
    detector.test()