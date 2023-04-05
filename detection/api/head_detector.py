from detector import Detector
import mediapipe as mp
import numpy as np
import cv2 

class HeadDetector(Detector):
    def __init__(self, consecutive_frames_threshold, landmarks_model_path, head_ratio_threshold):
        self.landmarks_model_path = landmarks_model_path
        self.dlib_facelandmark = dlib.shape_predictor(self.landmarks_model_path)
        self.consecutive_frames_threshold = consecutive_frames_threshold
        self.eye_ratio_threshold = head_ratio_threshold
        self.mp_pose = mp.solutions.pose
        self.frames = []
    
    def calculate_head_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]- b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def calculate_head_inclination(a, b):
        a = np.array(a)
        b = np.array(b)

        radians = np.arctan2(b[1]-a[1], b[0]-a[0])
        angle = np.degrees(radians)

        if angle < 0:
            angle = angle * -1

        return angle
    
    def head_angle(self, landmarks, image):
        r_ear = [landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y]
        nose = [landmarks[self.mp_pose.PoseLandmark.NOSE.value].x, landmarks[self.mp_pose.PoseLandmark.NOSE.value].y]
        l_ear = [landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y]

        angle = self.calculate_head_angle(r_ear, nose, l_ear)
        print(angle)
        
        if angle > 110:
            cv2.putText(image, f"HEAD ANGLE: {angle:.2f}", 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, f"HEAD ANGLE: {angle:.2f}", 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    def head_inclination(self, landmarks, image):
        l_ear = [landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y]
        r_ear = [landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y]

        angle = self.calculate_head_inclination(r_ear, l_ear)

        if angle <= 30:
            cv2.putText(image, f"HEAD INCLINATION: {angle:.2f}", 
                (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        elif 30 < angle < 50:
            cv2.putText(image, f"HEAD INCLINATION: {angle:.2f}", 
                (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, f"HEAD INCLINATION: {angle:.2f}", 
                (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            
    def execute(self, source):
        return super().execute(source)