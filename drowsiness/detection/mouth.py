from functools import partial

import numpy as np
import cv2 as cv

from drowsiness.detection.detector import DetectionData

OUTER_LIP = np.array([52, 64, 63, 67, 68, 61, 58, 59, 53, 56, 55])
INNER_LIP = np.array([65, 66, 62, 70, 69, 57, 60, 54])

WEIGHTS = {
    "yawn_count_weight": 0.2,
    "yawn_percentage_weight": 0.4,
    "yawn_time_weight": 0.4,
}


def inner_lip_area(landmarks):
    inner_lip = np.array(landmarks[INNER_LIP])

    area = cv.contourArea(inner_lip)

    return area


def execute(landmarks, fps=24, video_length=30, yawn_area=300, yawn_duration=4):
    frame_length = 1 / fps

    detection_data = {"yawn_count": 0, "yawn_frame_count": 0}

    area_array = []
    yawn_frames = 0
    for landmark in landmarks:
        inner_area = inner_lip_area(landmark)

        if inner_area > yawn_area:
            yawn_frames += 1
        else:
            if yawn_frames > yawn_duration:
                detection_data["yawn_count"] += 1
                detection_data["yawn_frame_count"] += yawn_frames
            yawn_frames = 0

        area_array.append(inner_area)

    detection_data["yawn_count"] /= 10  # max_num_yawn
    detection_data["yawn_percentage"] = detection_data["yawn_frame_count"] / len(
        landmarks
    )
    detection_data["yawn_time"] = (
        detection_data["yawn_frame_count"] * frame_length
    ) / video_length

    result = (
        (detection_data["yawn_count"] * WEIGHTS["yawn_count_weight"])
        + (detection_data["yawn_percentage"] * WEIGHTS["yawn_percentage_weight"])
        + (detection_data["yawn_time"] * WEIGHTS["yawn_time_weight"])
    )

    return DetectionData(round(result, 1), detection_data)


if __name__ == "__main__":
    config = {}

    test_execute = partial(execute, **config)
