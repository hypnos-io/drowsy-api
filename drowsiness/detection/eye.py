import glob
from functools import partial

import numpy as np
import cv2 as cv

from drowsiness.detection.detector import DetectionData


def create_frame_list():
    images = glob.glob(r"drowsy-api\detection\testing\frames\test\*.png")

    # Apply image processing techniques
    frames = [cv.imread(image) for image in images]
    frames = [
        cv.resize(frame, (640, 360)) for frame in frames
    ]  # resize images to a standard size
    frames = [cv.cvtColor(frame, cv.COLOR_RGB2BGR) for frame in frames]

    # Apply camera calibration
    camera_matrix = np.array(
        [[1000, 0, 320], [0, 1000, 180], [0, 0, 1]]
    )  # example camera matrix

    distortion_coeffs = np.array([0.1, -0.05, 0, 0])  # example distortion coefficients
    frames = [cv.undistort(frame, camera_matrix, distortion_coeffs) for frame in frames]

    return frames


# [0]P1 [1]P2 [2]P3 [3]P4 [4]P5 [5]P6
LEFT_EYE = np.array([35, 41, 42, 39, 37, 36])
RIGHT_EYE = np.array([89, 95, 96, 93, 91, 90])

WEIGHTS = {
    "eye_opening_weight": 0.4,
    "blink_count_weight": 0.2,
    "close_eyes_time_weight": 0.4,
}
BLINK_MAX = 15


def calculate_ear(eye):
    # eye: [0]P1 [1]P2 [2]P3 [3]P4 [4]P5 [5]P6
    """EAR = (|P2 - P6| + |P3 - P5|) / (2 * |P1 - P4|)"""
    vertical_distl = np.linalg.norm(eye[1] - eye[-1])
    vertical_distr = np.linalg.norm(eye[2] - eye[-2])
    horizontal_dist = np.linalg.norm(eye[0] - eye[3])
    eye_aspect_ratio = (vertical_distl + vertical_distr) / (2.0 * horizontal_dist)

    return eye_aspect_ratio


def average_ear(landmarks):
    if landmarks is None:
        return None
    left_eye = np.array(landmarks[LEFT_EYE])
    right_eye = np.array(landmarks[RIGHT_EYE])

    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)
    average = np.mean((left_ear, right_ear))

    return average


def execute(
    landmarks, ear_threshold=0.17, closed_eyes_threshold=3, fps=24, video_length=30
) -> DetectionData:
    if landmarks is None:
        print("Lista vazia.")
        return
    frame_length = 1 / fps

    detection_data = {
        "eye_opening": 0,
        "blink_count": 0,
        "closed_eyes_time": 0,
    }

    ear_list = []
    close_frames = 0
    for landmark in landmarks:
        ear = average_ear(landmark)
        if ear is not None:
            ear_list.append(ear)
            if ear < ear_threshold:
                close_frames += 1
                detection_data["closed_eyes_time"] += 1
            else:
                if 1 <= close_frames < closed_eyes_threshold:
                    detection_data["blink_count"] += 1
                close_frames = 0

    if not ear_list:
        return DetectionData(
            0, {"blink_count": 0, "eye_opening": 0, "closed_eyes_time": 0}
        )

    ear_array = np.array(ear_list)
    ear_array = ear_array[~np.isnan(ear_array)]
    ear_min = np.min(ear_array)
    ear_max = np.max(ear_array)
    ear_norm = abs(ear_array - ear_max) / abs(ear_min - ear_max)
    average = np.mean(ear_norm)
    video_length = len(ear_norm) / 10

    detection_data["eye_opening"] = average
    detection_data["closed_eyes_time"] = (
        detection_data["closed_eyes_time"] * frame_length
    ) / video_length
    detection_data["blink_count"] /= 20  # Blink max

    result = (
        detection_data["eye_opening"] * WEIGHTS["eye_opening_weight"]
        + detection_data["closed_eyes_time"] * WEIGHTS["close_eyes_time_weight"]
        + detection_data["blink_count"] * WEIGHTS["blink_count_weight"]
    )

    return DetectionData(round(result, 1), detection_data)


if __name__ == "__main__":
    config = {
        "ear_threshold": 0.17,
        "closed_eyes_threshold": 3,
        "fps": 10,
        "video_length": 32,
    }
    test_execute = partial(execute, **config)

    frame_sequence = create_frame_list()
    if len(frame_sequence) <= 0:
        print("Lista vazia.")
    else:
        print("\nRunning detection....")
        response = test_execute(frame_sequence)
        print("=" * 30 + " RESULTS: " + 30 * "=")
        print(response.data)
        print(response.result)
        if response.result < 0.4:
            print("Not tired")
        elif 0.4 <= response.result < 0.7:
            print("Kinda tired")
        else:
            print("Tired")
