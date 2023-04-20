import glob
from detector import AbstractDetector
import mediapipe as mp
import numpy as np
import itertools
import cv2 as cv
from time import time
import sys

def load_image(image):
    return cv.imread(image, v.IMREAD_GRAYSCALE)


def create_frame_list(location, extension):
        images = glob.glob(f"detection/api/frames/{location}/*.{extension}")
        
        frames = [cv.imread(image) for image in images]
        
        frames = [cv.cvtColor(frame, cv.COLOR_BGR2RGB) for frame in frames]

        return frames

class HeadDetector(AbstractDetector):
    def __init__(self, head_down_threshold=70, fps=60, eye_ratio_threshold=0.22):
        self.head_down_threshold = head_down_threshold
        self.eye_ratio_threshold = eye_ratio_threshold
        self.frames = []
        self.fps = fps
        self.mp_pose = mp.solutions.pose
        self.pose_images = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def __get_head__(self, results, frame):
        RIGHT_EAR_INDEXES = list(set(itertools.chain(*self.mp_pose.PoseLandmark.RIGHT_EAR)))
        LEFT_EAR_INDEXES = list(set(itertools.chain(*self.mp_pose.PoseLandmark.LEFT_EAR)))
        NOSE_INDEXES = list(set(itertools.chain(*self.mp_pose.PoseLandmark.NOSE)))

        right_ear_landmarks = [results.pose_landmarks.landmark[i] for i in RIGHT_EAR_INDEXES]
        left_ear_landmarks = [results.pose_landmarks.landmark[i] for i in LEFT_EAR_INDEXES]
        nose_landmarks = [results.pose_landmarks.landmark[i] for i in NOSE_INDEXES]

        height, width, _ = frame.shape
        right_ear_positions = [(int(l.x * width), int(l.y * height)) for l in right_ear_landmarks]
        left_ear_positions = [(int(l.x * width), int(l.y * height)) for l in left_ear_landmarks]
        nose_positions = [(int(l.x * width), int(l.y * height)) for l in nose_landmarks]

        return right_ear_positions, left_ear_positions, nose_positions
    
    def __calculate_head_angle__(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
            a[1] - b[1], a[0] - b[0]
        )
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def __calculate_head_inclination__(self, a, b):
        a = np.array(a)
        b = np.array(b)

        radians = np.arctan2(b[1] - a[1], b[0] - a[0])
        angle = np.degrees(radians)

        if angle < 0:
            angle = angle * -1

        return angle
    
    def __detect__(
            self, 
            frames,
            angle_threshold=110, 
            inclination_side_threshold=40, 
            consec_frames_threshold_angle=2,
            consec_side_threshold_angle=2):

        # HEAD INCLINATION 
        head_inclination_up_time = 0
        head_inclination_down_time = 0
        consecutive_inclination_down_frames = 0
        total_inclination_down_time = 0
        head_inclination_mean = 0

        # HEAD ANGLE 
        head_angle_up_time = 0 
        head_angle_down_time = 0
        consecutive_angle_down_frames = 0
        total_angle_down_time = 0
        head_angle_mean = 0

        fr = 0

        for frame in frames:

            fr += 1

            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = self.pose_images.process(image)

            image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            if results.pose_landmarks:
                for pose_landmarks in results.pose_landmarks:
                    right_ear_pos, left_ear_pos, nose_pos = self.__get_eyes__(results, frame)
                    head_angle = self.__calculate_head_angle__(right_ear_pos, nose_pos, left_ear_pos)
                    head_inclination = self.__calculate_head_inclination__(right_ear_pos, left_ear_pos)
                    print("=-=-=-=-=-=-=-=")
                    print("Right Ear Pos: " + right_ear_pos)
                    print("Left Ear Pos: " + left_ear_pos)
                    print("Nose Pos: " + nose_pos)

                    # Head Angle
                    if head_angle > angle_threshold:
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

                    # Head Inclination
                    if head_inclination <= inclination_side_threshold:
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

        
        total_angle_down_time = total_angle_down_time / self.fps
        total_inclination_down_time = total_inclination_down_time / self.fps
        head_angle_mean = head_angle_mean/len(frames)
        head_inclination_mean = head_inclination_mean/len(frames)
        return total_angle_down_time, total_inclination_down_time, head_angle_mean, head_inclination_mean
            
    # def __head_detection_old__(
    #         self, 
    #         frames, 
    #         angle_threshold=110, 
    #         inclination_side_threshold=40, 
    #         consec_frames_threshold_angle=2,
    #         consec_side_threshold_angle=2):

    #     with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:


    #         head_inclination_up_time = 0
    #         head_inclination_down_time = 0
    #         consecutive_inclination_down_frames = 0
    #         total_inclination_down_time = 0
    #         head_inclination_mean = 0

    #         # HEAD ANGLE (UP AND DOWN)
    #         head_angle_up_time = 0 
    #         head_angle_down_time = 0
    #         consecutive_angle_down_frames = 0
    #         total_angle_down_time = 0
    #         head_angle_mean = 0

    #         for frame in frames:
    #             image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #             image.flags.writeable = False

    #             results = pose.process(image)

    #             image.flags.writeable = True
    #             image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    #             landmarks = results.pose_landmarks.landmark

    #             # HEAD ANGLE
    #             r_ear = [
    #                 landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x,
    #                 landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y,
    #             ]
    #             nose = [
    #                 landmarks[self.mp_pose.PoseLandmark.NOSE.value].x,
    #                 landmarks[self.mp_pose.PoseLandmark.NOSE.value].y,
    #             ]
    #             l_ear = [
    #                 landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x,
    #                 landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y,
    #             ]

    #             angle = self.calculate_head_angle(r_ear, nose, l_ear)
    #             head_angle_mean += angle
                
    #             if angle > angle_threshold:
    #                 consecutive_angle_down_frames += 1

    #                 if consecutive_angle_down_frames == consec_frames_threshold_angle:
    #                     head_angle_down_time += 1 / len(frames)
    #                     total_angle_down_time += head_angle_down_time
    #                     head_angle_up_time = 0

    #             else:
    #                 consecutive_angle_down_frames = 0

    #                 if head_angle_up_time > 0:
    #                     head_angle_down_time += 1 / len(frames)
    #                     if head_angle_down_time > 1.0:
    #                         head_angle_down_time = 0
    #                         head_angle_up_time = 0

    #             # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-

    #             # HEAD INCLINATION
    #             l_ear = [
    #                 landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x,
    #                 landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y,
    #             ]
    #             r_ear = [
    #                 landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x,
    #                 landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y,
    #             ]

    #             angle = self.__calculate_head_inclination__(r_ear, l_ear)
    #             head_inclination_mean += angle

    #             if angle <= inclination_side_threshold:
    #                 consecutive_inclination_down_frames += 1

    #                 if consecutive_inclination_down_frames == consec_side_threshold_angle:
    #                     head_inclination_down_time += 1 / len(frames)
    #                     total_inclination_down_time += head_inclination_down_time
    #                     head_inclination_up_time = 0
    #             else:
    #                 consecutive_inclination_down_frames = 0
                    
    #                 if head_inclination_up_time > 0:
    #                     head_inclination_down_time += 1 / len(frames)
    #                     if head_inclination_down_time > 1.0:
    #                         head_inclination_down_time = 0
    #                         head_inclination_up_time = 0


    #     total_angle_down_time = total_angle_down_time / self.fps
    #     total_inclination_down_time = total_inclination_down_time / self.fps
    #     head_angle_mean = head_angle_mean/len(frames)
    #     head_inclination_mean = head_inclination_mean/len(frames)

    #     return total_angle_down_time, total_inclination_down_time, head_angle_mean, head_inclination_mean


    def execute(self, frames):
        if len(frames) <= 0:
            raise ValueError("Lista de frames vazia")
        "Executes the Head detection"
        TADT, TIDT, HAM, HIM = self.__detect__(frames)
        result_dict = {
            "Total Head Angle Time": round(TADT, 4),
            "Total Head Inclination Time": round(TIDT, 4),
            "Head Angle Mean": round(HAM, 2),
            "Head Inclination Mean": round(HIM, 2)
        }

        return result_dict

if __name__ == "__main__":

    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a camera")
        exit()

    detector = HeadDetector()
    prev = 0
    capture = True
    while capture:
        time_elapsed = time() - prev
        ret, frame = cap.read()

        if not ret:
            print("Não foi possivel capturar imagens da camera. Encerrando execução.")
            break

        key = cv.waitKey(1)

        if key == ord("q"):
            cap.release()
            capture = False

        if time_elapsed > 1.0 / detector.fps:
            prev = time()

            data = detector.execute(frame)

            y = 20
            for key, value in data.items():
                cv.putText(
                    frame,
                    f"{key}: {value:.2f}",
                    (10, y),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    1,
                    2,
                )
                y += 20

            cv.imshow("frame", frame)

    cv.destroyAllWindows()
