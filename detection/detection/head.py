import glob
from time import time

from detection.detector import DetectionData, MediapipeDetector
import cv2 as cv
import numpy as np

RIGHT_EAR = MediapipeDetector["pose"].PoseLandmark.RIGHT_EAR
LEFT_EAR = MediapipeDetector["pose"].PoseLandmark.LEFT_EAR
NOSE = MediapipeDetector["pose"].PoseLandmark.NOSE

WEIGHTS = {
    "frontal_angle_mean_weight": 0.2,
    "frontal_down_time_weight": 0.8,
    "lateral_angle_mean_weight": 0.8,
    "lateral_down_time_weight": 0.2,
    "frontal_down_count_weight": 0.5,
    "lateral_down_count_weight": 0.5,
    "frontal_weight": 0.8,
    "lateral_weight": 0.2,
}


def create_frame_list(extension):
    images = glob.glob(
        f"C:/Users/callidus/drowsy-api/detection/detection/testing/frames/*.{extension}"
    )

    frames = [cv.imread(image) for image in images]

    frames = [cv.cvtColor(frame, cv.COLOR_BGR2RGB) for frame in frames]

    return frames


def get_head(results, height, width):
    right_ear_landmarks = results.pose_landmarks.landmark[RIGHT_EAR]
    left_ear_landmarks = results.pose_landmarks.landmark[LEFT_EAR]
    nose_landmarks = results.pose_landmarks.landmark[NOSE]

    right_ear_positions = (
        int(right_ear_landmarks.x * width),
        int(right_ear_landmarks.y * height),
    )
    left_ear_positions = (
        int(left_ear_landmarks.x * width),
        int(left_ear_landmarks.y * height),
    )
    nose_positions = (int(nose_landmarks.x * width), int(nose_landmarks.y * height))

    return right_ear_positions, left_ear_positions, nose_positions


def calculate_head_frontal(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    if b[1] < a[1] - 17 or b[1] < c[1] - 17:
        angle = 140

    return angle


def calculate_head_lateral(a, b):
    a = np.array(a)
    b = np.array(b)

    radians = np.arctan2(b[1] - a[1], b[0] - a[0])
    angle = np.degrees(radians)

    if angle < 0:
        angle = angle * -1

    return angle


def calculate_angles(shape, results):
    height, width, *_ = shape

    data = {"head_frontal": 0, "head_lateral": 0}

    if results.pose_landmarks:
        right_ear_pos, left_ear_pos, nose_pos = get_head(results, height, width)

        head_frontal = calculate_head_frontal(right_ear_pos, nose_pos, left_ear_pos)
        head_lateral = calculate_head_lateral(right_ear_pos, left_ear_pos)

        data["head_frontal"] = head_frontal
        data["head_lateral"] = head_lateral
    else:
        return None

    return data


def execute(
    landmarks,
    shape,
    consec_frames_threshold_frontal=2,
    consec_frames_threshold_lateral=2,
    fps=24,
    frontal_threshold=120,
    lateral_threshold=20,
    video_length=30,
):
    frame_length = 1 / fps

    detection_data = {
        "total_frontal_down_time": 0,
        "head_frontal_angle_mean": 0,
        "total_lateral_down_time": 0,
        "head_lateral_angle_mean": 0,
        "total_frontal_down_count": 0,
        "total_lateral_down_count": 0,
    }

    lateral_down_count = 0
    frontal_down_count = 0

    frontal_down_consecutives = 0
    lateral_down_consecutives = 0

    frame_data = []

    frames = 0
    for result in landmarks:
        frames += 1
        data = calculate_angles(shape, result)

        if data is not None:
            # Head frontal
            if data["head_frontal"] < frontal_threshold:
                frontal_down_consecutives += 1

                if frontal_down_consecutives < 2:
                    detection_data["total_frontal_down_count"] += 1

                if frontal_down_consecutives > consec_frames_threshold_frontal:
                    print(
                        f"\033[31m Frontal :{frames}\033[0m | Angle: {data['head_frontal']}"
                    )
                    frontal_down_count += 1

            else:
                frontal_down_consecutives = 0

            # # Head lateral
            if data["head_lateral"] > lateral_threshold:
                lateral_down_consecutives += 1

                if frontal_down_consecutives < 2:
                    detection_data["total_lateral_down_count"] += 1

                if lateral_down_consecutives > consec_frames_threshold_lateral:
                    print(
                        f"\033[32m Lateral :{frames}\033[0m | Angle: {data['head_lateral']}"
                    )
                    lateral_down_count += 1

            else:
                lateral_down_consecutives = 0

            frame_data.append(data)

    frontal_angle_list = np.array(
        [data_frontal["head_frontal"] for data_frontal in frame_data]
    )
    frontal_norm = (frontal_angle_list - np.max(frontal_angle_list)) / (
        np.min(frontal_angle_list) - np.max(frontal_angle_list)
    )

    detection_data["head_frontal_angle_mean"] = np.mean(frontal_norm)

    detection_data["total_frontal_down_time"] = (
        frontal_down_count * frame_length
    ) / video_length
    detection_data["total_frontal_down_count"] /= 20  # Down max

    lateral_angle_list = np.array(
        [data_lateral["head_lateral"] for data_lateral in frame_data]
    )
    lateral_norm = (lateral_angle_list - np.min(lateral_angle_list)) / (
        np.max(lateral_angle_list) - np.min(lateral_angle_list)
    )

    detection_data["head_lateral_angle_mean"] = np.mean(lateral_norm)

    detection_data["total_lateral_down_time"] = (
        lateral_down_count * frame_length
    ) / video_length
    detection_data["total_lateral_down_count"] /= 20  # Down max

    # Result
    final_result_frontal = (
        detection_data["head_frontal_angle_mean"] * WEIGHTS["frontal_angle_mean_weight"]
        + detection_data["total_frontal_down_time"]
        * WEIGHTS["frontal_down_time_weight"]
        + detection_data["total_frontal_down_count"]
        * WEIGHTS["frontal_down_count_weight"]
    )

    final_result_lateral = (
        detection_data["head_lateral_angle_mean"] * WEIGHTS["lateral_angle_mean_weight"]
        + detection_data["total_lateral_down_time"]
        * WEIGHTS["lateral_down_time_weight"]
        + detection_data["total_lateral_down_count"]
        * WEIGHTS["lateral_down_count_weight"]
    )

    result = (
        (final_result_frontal * WEIGHTS["frontal_weight"])
        + final_result_lateral * WEIGHTS["lateral_weight"]
    ) / 2

    return DetectionData(round(result, 1), detection_data)


if __name__ == "__main__":
    print("")
