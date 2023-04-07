import cv2
import dlib
import os
import glob
from scipy.spatial import distance
import numpy as np
from detector import Detector
from multiprocessing import Pool

def load_image(image):
    return cv2.imread(image, cv2.IMREAD_GRAYSCALE)


def create_frame_list(local):
        #images = glob.glob(f"detection/api/frames/{local}/*.png")
        images = glob.glob("detection/api/frames/closed_eye.jpg")
        frames = [cv2.imread(image, cv2.IMREAD_GRAYSCALE) for image in images]
            
        return frames


class EyeDetector(Detector):
    def __init__(self, consecutive_frames_threshold, landmarks_model_path, eye_ratio_threshold=0.20):
        self.landmarks_model_path = landmarks_model_path
        self.dlib_facelandmark = dlib.shape_predictor(self.landmarks_model_path)
        self.consecutive_frames_threshold = consecutive_frames_threshold
        self.eye_ratio_threshold = eye_ratio_threshold
        self.hog_face_detector = dlib.get_frontal_face_detector()
        self.frames = []
        
    def calculate_ear(self, eye):
        eye = np.array(eye)
        p2_minus_p6 = np.linalg.norm(eye[1] - eye[5])
        p3_minus_p5 = np.linalg.norm(eye[2] - eye[4])
        p1_minus_p4 = np.linalg.norm(eye[0] - eye[3])

        ear_aspect_ratio = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
        return ear_aspect_ratio
    
    def get_eyes(self, frame, face):
        face_landmarks = self.dlib_facelandmark(frame, face)
        left_eye = []
        right_eye = []

        for n in range(36,42):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	left_eye.append((x,y))

        for n in range(42,48):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	right_eye.append((x,y))
         
        return left_eye, right_eye
        
    
    def __closed_eyes__(self, consecutive_frames_threshold_closed, frames):
        if len(frames) <= 0:
            raise ValueError("Lista de frames vazia. Por favor, verifique.")

        eye_closed_time = 0
        eye_opened_time = 0
        consecutive_frames = 0
        total_eye_closed_time = 0
        blink_count = 0
        blink_threshold = 2
        ear_mean = 0
        
        for frame in frames:
            faces = self.hog_face_detector(frame)
            
            for face in faces:
                left_eye, right_eye = self.get_eyes(frame, face)
                
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                
                EAR = (left_ear+right_ear)/2
                EAR = round(EAR,2)
                ear_mean += EAR
                
                if EAR < self.eye_ratio_threshold:
                    consecutive_frames += 1
                    
                    if consecutive_frames >= consecutive_frames_threshold_closed:
                        eye_closed_time += 1 / len(frames)
                        total_eye_closed_time += eye_closed_time
                        eye_opened_time = 0
                    
                else:
                    if consecutive_frames >= blink_threshold:
                        blink_count += 1
                    consecutive_frames = 0
                    
                    if eye_closed_time > 0:
                        eye_opened_time += 1 / len(frames)
                        if eye_opened_time > 1.0:
                            eye_closed_time = 0
                            eye_opened_time = 0
                            
        ear_mean = ear_mean/len(frames)
        return total_eye_closed_time, blink_count, ear_mean
    
    def execute(self, frames):
        "Executes the Eye detection"
        time, blink, ear = self.__closed_eyes__(3, frames)
        detection_dict = {
            "Piscadas": blink,
            "Olhos fechados": round(time, 4),
            "Abertura dos olhos": round(ear, 2)
        }
        
        return detection_dict
