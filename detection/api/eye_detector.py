import cv2
import dlib
import os
import glob
from scipy.spatial import distance
import numpy as np
from api.detector import Detector
from classification.detection_data import DetectionData
from multiprocessing import Pool

def load_image(image):
    return cv2.imread(image, cv2.IMREAD_GRAYSCALE)


# def create_frame_list(local, extension):
#         images = glob.glob(f"detection/api/frames/{local}/*.{extension}")
#         frames = [cv2.imread(image, cv2.IMREAD_GRAYSCALE) for image in images]
            
#         return frames

def create_frame_list(location, extension):
        images = glob.glob(f"detection/api/frames/{location}/*.{extension}")
        
        frames = [cv2.imread(image) for image in images]
        
        frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

        return frames


class EyeDetector(Detector):
    def __init__(self, closed_eyes_threshold, blink_threshold, landmarks_model_path, fps=10, eye_ratio_threshold=0.20):
        self.landmarks_model_path = f"{landmarks_model_path}/shape_predictor_68_face_landmarks.dat"
        self.dlib_facelandmark = dlib.shape_predictor(self.landmarks_model_path)
        self.closed_eyes_threshold = closed_eyes_threshold
        self.eye_ratio_threshold = eye_ratio_threshold
        self.hog_face_detector = dlib.get_frontal_face_detector()
        self.frames = []
        self.fps = fps
        self.blink_threshold = blink_threshold
        
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
        
    
    def __detect__(self, frames):
        if len(frames) <= 0:
            raise ValueError("Lista de frames vazia. Por favor, verifique.")

        eye_closed_time = 0
        eye_opened_time = 0
        consecutive_frames = 0
        total_eye_closed_time = 0
        blink_count = 0
        ear_mean = 0
        fr = 0
        
        for frame in frames:
            faces = self.hog_face_detector(frame)
            fr += 1
            
            for face in faces:
                left_eye, right_eye = self.get_eyes(frame, face)
                
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                
                EAR = (left_ear+right_ear)/2
                EAR = round(EAR,2)
                ear_mean += EAR
                print(f"FRAME {fr} | EAR: {EAR}")
                
                if EAR < self.eye_ratio_threshold:
                    consecutive_frames += 1
                    print(f":{consecutive_frames}")
                    
                    if consecutive_frames >= self.closed_eyes_threshold:
                        print(f"eyes closed at frame {fr}")
                        eye_closed_time += 1 / self.fps
                        total_eye_closed_time += eye_closed_time
                        eye_opened_time = 0
                    
                else:
                    if consecutive_frames >= self.blink_threshold:
                        blink_count += 1
                    consecutive_frames = 0
                    
                    if eye_closed_time > 0:
                        eye_opened_time += 1 / self.fps
                        if eye_opened_time > 1.0:
                            eye_closed_time = 0
                            eye_opened_time = 0
                            
        ear_mean = ear_mean/len(frames)
        return total_eye_closed_time, blink_count, ear_mean
    
    def execute(self, frames):
        """Executes the Eye detection"""
        if len(frames) <= 0:
            raise ValueError("Lista de frames vazia")
        time, blink, ear = self.__detect__(frames)
        detection_dict = {
            "blinks": blink,
            "closed_eyes": round(time, 4),
            "eye_opening": round(ear, 2)
        }
        
        data = DetectionData(0, detection_dict)
        
        return data
