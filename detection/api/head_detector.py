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

    def head_inclination(
            self, 
            landmarks,  
            head_soft_count,
            head_hard_count,
            inclination_middle_threshold=30, 
            inclination_bottom_threshold=50):
        l_ear = [landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y]
        r_ear = [landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y]

        angle = self.calculate_head_inclination(r_ear, l_ear)

        # HEAD INCLINATION DETECTIONS
        if angle <= inclination_middle_threshold:
            head_soft_count += 1
        elif inclination_middle_threshold < angle < inclination_bottom_threshold:
            head_hard_count += 1
        else:
            pass
            
        return head_soft_count, head_soft_count
            
    def head_detection(
            self, 
            frames, 
            angle_threshold=110, 
            inclination_pos1_threshold=30, 
            inclination_pos2_threshold=50,
            consec_frames_threshold_angle=2,
            consec_pos1_threshold_angle=2,
            consec_pos2_threshold_angle=2):

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

            # HEAD INCLINATION (POS 0, 1, 2)
            head_pos0_time = 0

            head_pos1_time = 0
            consecutive_pos1_frames = 0
            total_pos1_time = 0
            pos1_threshold = 2

            head_pos2_time = 0
            consecutive_pos2_frames = 0
            total_pos2_time = 0
            pos2_threshold = 2


            # HEAD ANGLE (UP AND DOWN)
            head_down_time = 0
            head_up_time = 0 
            consecutive_angle_frames = 0
            total_head_angle_time = 0

            for frame in frames:
                image  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = pose.process(image)
                
                image.flags.writeable = True
                image  = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


                landmarks = results.pose_landmarks.landmark

                # HEAD ANGLE
                r_ear = [landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y]
                nose = [landmarks[self.mp_pose.PoseLandmark.NOSE.value].x, landmarks[self.mp_pose.PoseLandmark.NOSE.value].y]
                l_ear = [landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y]

                angle = self.calculate_head_angle(r_ear, nose, l_ear)
                print(angle)
                
                if angle > angle_threshold:
                    consecutive_angle_frames += 1

                    if consecutive_angle_frames == consec_frames_threshold_angle:
                        head_down_time += 1 / len(frames)
                        total_head_angle_time += head_down_time
                        head_up_time = 0

                else:
                    consecutive_angle_frames = 0

                    if head_up_time > 0:
                        head_down_time += 1 / len(frames)
                        if head_down_time > 1.0:
                            head_down_time = 0
                            head_up_time = 0

                # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-

                # HEAD INCLINATION
                l_ear = [landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y]
                r_ear = [landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y]

                angle = self.calculate_head_inclination(r_ear, l_ear)

                if angle <= inclination_pos1_threshold:
                    consecutive_pos1_frames += 1

                    if consecutive_pos1_frames == consec_pos1_threshold_angle:
                        head_pos1_time += 1 / len(frames)
                        total_pos1_time += head_pos1_time
                        head_pos0_time = 0

                elif inclination_pos1_threshold < angle < inclination_pos2_threshold:
                    consecutive_pos2_frames += 1

                    if consecutive_pos2_frames == consec_pos2_threshold_angle:
                        head_pos2_time += 1 / len(frames)
                        total_pos2_time += head_pos2_time
                        head_pos0_time = 0
                else:
                    consecutive_pos1_frames = 0
                    consecutive_pos2_frames = 0
                    
                    if head_pos0_time > 0:
                        head_pos1_time += 1 / len(frames)
                        head_pos2_time += 1 / len(frames)

                        if head_pos0_time > 1.0:
                            head_pos0_time = 0
                            head_pos1_time = 0
                            head_pos2_time = 0

        return total_head_angle_time, total_pos1_time, total_pos2_time, 



    def cropROI(self, source):
        pass

    def execute(self, frames):
        "Executes the Head detection"
        time, blink, ear = self.head_detection(3, frames)