from detector import Detector
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
            
    def head_detection(
            self, 
            frames, 
            angle_threshold=110, 
            inclination_side_threshold=40, 
            consec_frames_threshold_angle=2,
            consec_side_threshold_angle=2):

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:


            head_inclination_up_time = 0
            head_inclination_down_time = 0
            consecutive_inclination_down_frames = 0
            total_inclination_down_time = 0

            # HEAD ANGLE (UP AND DOWN)
            head_angle_up_time = 0 
            head_angle_down_time = 0
            consecutive_angle_down_frames = 0
            total_angle_down_time = 0

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
                
                if angle > angle_threshold:
                    consecutive_angle_down_frames += 1

                    if consecutive_angle_down_frames == consec_frames_threshold_angle:
                        head_angle_down_time += 1 / len(frames)
                        total_angle_down_time += head_angle_down_time
                        head_angle_up_time = 0

                else:
                    consecutive_angle_down_frames = 0

                    if head_angle_up_time > 0:
                        head_angle_down_time += 1 / len(frames)
                        if head_angle_down_time > 1.0:
                            head_angle_down_time = 0
                            head_angle_up_time = 0

                # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-

                # HEAD INCLINATION
                l_ear = [landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y]
                r_ear = [landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y]

                angle = self.calculate_head_inclination(r_ear, l_ear)

                if angle <= inclination_side_threshold:
                    consecutive_inclination_down_frames += 1

                    if consecutive_inclination_down_frames == consec_side_threshold_angle:
                        head_inclination_down_time += 1 / len(frames)
                        total_inclination_down_time += head_inclination_down_time
                        head_inclination_up_time = 0
                else:
                    consecutive_inclination_down_frames = 0
                    
                    if head_inclination_up_time > 0:
                        head_inclination_down_time += 1 / len(frames)
                        if head_inclination_down_time > 1.0:
                            head_inclination_down_time = 0
                            head_inclination_up_time = 0

        return total_angle_down_time, total_inclination_down_time



    def cropROI(self, source):
        pass

    def execute(self, frames):
        "Executes the Head detection"
        total_angle_down_time, total_inclination_down_time = self.head_detection(frames)
        result_dict = {
            "Total Head Angle Time": total_angle_down_time,
            "Total Head Inclination Time": total_inclination_down_time,
        }
        
        return result_dict