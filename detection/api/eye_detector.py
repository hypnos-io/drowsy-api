import cv2
import dlib
import os
import glob
from scipy.spatial import distance
from detector import Detector


def create_frame_list():
        images = glob.glob("detection/api/frames/*.png")

        frames = []

        for image in images:
            img = cv2.imread(image)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            frames.append(gray)
            
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
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear_aspect_ratio = (A+B)/(2.0*C)
        return ear_aspect_ratio
    
    def get_eyes(self, frame, face):
        face_landmarks = self.dlib_facelandmark(frame, face)
        left_eye = []
        right_eye = []

        for n in range(36,42):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	left_eye.append((x,y))
        	next_point = n+1
        	if n == 41:
        		next_point = 36
        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y
        	cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        for n in range(42,48):
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	right_eye.append((x,y))
        	next_point = n+1
        	if n == 47:
        		next_point = 42
        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y
        	cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
         
        return left_eye, right_eye
        
    
    def closed_eyes(self, consecutive_frames_threshold, frames):
        eye_closed_time = 0
        eye_opened_time = 0
        consecutive_frames = 0
        total_eye_closed_time = 0
        
        for frame in frames:
            faces = self.hog_face_detector(frame)
            
            for face in faces:
                left_eye, right_eye = self.get_eyes(frame, face)
                
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                
                EAR = (left_ear+right_ear)/2
                EAR = round(EAR,2)
                
                if EAR < self.eye_ratio_threshold:
                    consecutive_frames += 1
                    
                    if consecutive_frames >= consecutive_frames_threshold:
                        eye_closed_time += 1 / len(frames)
                        total_eye_closed_time += eye_closed_time
                        eye_opened_time = 0
                else:
                    consecutive_frames = 0
                    
                    if eye_closed_time > 0:
                        eye_opened_time += 1 / len(frames)
                        if eye_opened_time > 1.0:
                            eye_closed_time = 0
                            eye_opened_time = 0
        
        return total_eye_closed_time

    def cropROI(self, source):
        pass
    "Executes the Eye detection"
    def execute(self, frames):
        pass
