from api.detector import Detector
import mediapipe as mp
import numpy as np
import cv2 

class HeadDetector(Detector):
    def __init__(self, head_ratio_threshold):
        self.head_ratio_threshold = head_ratio_threshold
        self.mp_pose = mp.solutions.pose
        self.frames = []
    
    def calculate_head_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]- b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def calculate_head_inclination(self, a, b):
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
            
        return head_soft_count, head_hard_count
            
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
            total_angle = []
            tilt = []

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
                total_angle.append(angle)
                #print(angle)
                
                if angle > angle_threshold:
                    consecutive_angle_frames += 1

                    if consecutive_angle_frames == consec_frames_threshold_angle:
                        head_down_time += 1 / 10
                        total_head_angle_time += head_down_time
                        head_up_time = 0

                else:
                    consecutive_angle_frames = 0

                    if head_up_time > 0:
                        head_down_time += 1 / 10
                        if head_down_time > 1.0:
                            head_down_time = 0
                            head_up_time = 0

                # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-

                # HEAD INCLINATION
                l_ear = [landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y]
                r_ear = [landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y]

                angle = self.calculate_head_inclination(r_ear, l_ear)
                tilt.append(angle)

                if angle <= inclination_pos1_threshold:
                    consecutive_pos1_frames += 1

                    if consecutive_pos1_frames == consec_pos1_threshold_angle:
                        head_pos1_time += 1 / 10
                        total_pos1_time += head_pos1_time
                        head_pos0_time = 0

                elif inclination_pos1_threshold < angle < inclination_pos2_threshold:
                    consecutive_pos2_frames += 1

                    if consecutive_pos2_frames == consec_pos2_threshold_angle:
                        head_pos2_time += 1 / 10
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
        avg_angle = round(np.mean(total_angle), 2)
        avg_tilt = round(np.mean(tilt),2)
        return round(total_head_angle_time, 2), avg_tilt



    def cropROI(self, source):
        pass

    def execute(self, frames):
        "Executes the head detection"
        total_head_angle_time, tilt = self.head_detection(frames)
        result_dict = {
            "total_head_angle_time": total_head_angle_time,
            "avegare tilt (L/R)": tilt,
        }
        
        return result_dict