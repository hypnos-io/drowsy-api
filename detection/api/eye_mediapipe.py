import os
import glob
from api.detector import Detector
import cv2
import itertools
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

def load_image(image):
    return cv2.imread(image, cv2.IMREAD_GRAYSCALE)


def create_frame_list(location, extension):
        images = glob.glob(f"detection/api/frames/{location}/*.{extension}")
        
        frames = [cv2.imread(image) for image in images]
        
        frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

        return frames


class EyeDetector(Detector):
    def __init__(self, closed_eyes_threshold, blink_threshold, fps=10, eye_ratio_threshold=0.22):
        self.closed_eyes_threshold = closed_eyes_threshold
        self.blink_threshold = blink_threshold
        self.eye_ratio_threshold = eye_ratio_threshold
        self.frames = []
        self.fps = fps
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh_images = self.mp_face_mesh.FaceMesh(max_num_faces=1)
        
    def __calculate_left_ear__(self, left_eye):
        """Calcula o EAR (Eye Aspect Ratio) do olho esquerdo utilizando a fórmula:
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
        
    
    def __detect__(self, frames):
        eye_closed_time = 0
        eye_opened_time = 0
        consecutive_frames = 0
        total_eye_closed_time = 0
        blink_count = 0
        ear_mean = 0
        cls_count = 0
        cnt = 0
        fr = 0
        for frame in frames:
            fr += 1
            results = self.face_mesh_images.process(frame[:,:,::-1])
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
            
                    left_eye_pos, right_eye_pos = self.__get_eyes__(results, frame)
                    left_eye_ear = self.__calculate_left_ear__(left_eye_pos)
                    right_eye_ear = self.__calculate_right_ear__(right_eye_pos)
                    EAR = (left_eye_ear + right_eye_ear) / 2
                    EAR = round(EAR, 2)
                    ear_mean += EAR
                    #print(f"FRAME {fr} | EAR: {EAR}")
                    
                    if EAR <= self.eye_ratio_threshold:
                        consecutive_frames += 1
                        #print(f":{consecutive_frames}")
                        
                        if consecutive_frames >= self.closed_eyes_threshold:
                            eye_closed_time += 1 / len(frames)  
                            total_eye_closed_time += 1
                            eye_opened_time = 0
                            cls_count += 1
                            #print(f"eye closed at frame {cls_count}")
                    else:
                        if consecutive_frames >= self.blink_threshold:
                                blink_count += 1
                        consecutive_frames = 0
                            
                        if eye_closed_time > 0:
                                eye_opened_time += 1 / len(frames)
                                if eye_opened_time > 1.0:
                                    eye_closed_time = 0
                                    eye_opened_time = 0
                                    
        ear_mean = ear_mean/len(frames)
        total_eye_closed_time = total_eye_closed_time / self.fps
        return total_eye_closed_time, blink_count, ear_mean
    
    def execute(self, frames):
        if len(frames) <= 0:
            raise ValueError("Lista de frames vazia")
        "Executes the Eye detection"
        time, blink, ear = self.__detect__(frames)
        detection_dict = {
            "blinks": blink,
            "closed_eyes": round(time, 4),
            "eye_opening": round(ear, 2)
        }
        
        return detection_dict
