import glob
import sys

sys.path.append(r"drowsiness")

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# from classification import KSSClassifier

from drowsiness.detection.detector import DetectionData, MediapipeDetector
from drowsiness.classification import KSSClassifier

RIGHT_EAR = 0
LEFT_EAR = 16
NOSE = 30



WEIGHTS = {
    "frontal_angle_mean_weight": 0.2,
    "frontal_down_time_weight": 0.2,
    "frontal_down_count_weight": 0.6,

    "lateral_angle_mean_weight": 0.3,
    "lateral_down_time_weight": 0.3,
    "lateral_down_count_weight": 0.4,
    
    "frontal_weight": 0.5,
    "lateral_weight": 0.5,
}


def create_frame_list():
    images = glob.glob(r"drowsiness\testing\frames\*.png")

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

def calculate_head_frontal(right, nose, left):
    radians = np.arctan2(left[1] - nose[1], left[0] - nose[0]) - np.arctan2(
        right[1] - nose[1], right[0] - nose[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    if nose[1] < right[1] - 17 or nose[1] < left[1] - 17:
        angle = 140

    return angle

def calculate_head_lateral(right, left):
    radians = np.arctan2(left[1] - right[1], left[0] - right[0])
    angle = np.degrees(radians)

    if angle < 0:
        angle = angle * -1

    return angle


def calculate_angles(landmarks):
    data = {"head_frontal": 0, "head_lateral": 0}

    if landmarks is not None:
        right_ear, left_ear, nose = landmarks[RIGHT_EAR], landmarks[LEFT_EAR], landmarks[NOSE]

        head_frontal = calculate_head_frontal(right_ear, nose, left_ear)
        head_lateral = calculate_head_lateral(right_ear, left_ear)

        data["head_frontal"] = head_frontal
        data["head_lateral"] = head_lateral
    else:
        return None

    return data


def execute(
    landmarks,
    consec_frames_threshold_frontal=0,
    consec_frames_threshold_lateral=0,
    fps=24,
    frontal_threshold=120,
    lateral_threshold=20,
):
    frame_length = 1 / fps

    detection_data = {
        "total_frontal_down_time": 0,
        "head_frontal_angle_mean": 0,
        "total_frontal_down_count": 0,
        "total_lateral_down_time": 0,
        "head_lateral_angle_mean": 0,
        "total_lateral_down_count": 0,
    }

    lateral_down_count = 0
    frontal_down_count = 0

    frontal_down_consecutives = 0
    lateral_down_consecutives = 0

    frame_data = []

    frames = 0
    for landmark in landmarks:
        frames += 1
        data = calculate_angles(landmark)

        if data is not None:
            frame_data.append(data)
            # Head frontal
            if data["head_frontal"] < frontal_threshold:
                frontal_down_consecutives += 1

                if frontal_down_consecutives < 2:
                    detection_data["total_frontal_down_count"] += 1

                if frontal_down_consecutives > consec_frames_threshold_frontal:
                    frontal_down_count += 1

            else:
                frontal_down_consecutives = 0

            # # Head lateral
            if data["head_lateral"] > lateral_threshold:
                lateral_down_consecutives += 1

                if frontal_down_consecutives < 2:
                    detection_data["total_lateral_down_count"] += 1

                if lateral_down_consecutives > consec_frames_threshold_lateral:
                    lateral_down_count += 1

            else:
                lateral_down_consecutives = 0

    video_length = frame_length * frames

    frontal_angle_list = np.array(
        [data_frontal["head_frontal"] for data_frontal in frame_data]
    )
    frontal_norm = (frontal_angle_list - np.max(frontal_angle_list)) / (
        60 - np.max(frontal_angle_list)
    )

    detection_data["head_frontal_angle_mean"] = np.mean(frontal_norm)

    detection_data["total_frontal_down_time"] = (
        frontal_down_count * frame_length
    ) / video_length
    detection_data["total_frontal_down_count"] /= 15 # Down max

    lateral_angle_list = np.array(
        [data_lateral["head_lateral"] for data_lateral in frame_data]
    )

    ceiling_value = np.max(lateral_angle_list)
    lateral_norm = abs(lateral_angle_list - ceiling_value) / abs(35 - ceiling_value)


    detection_data["head_lateral_angle_mean"] = np.mean(lateral_norm)


    detection_data["total_lateral_down_time"] = (
        lateral_down_count * frame_length
    ) / video_length

    detection_data["total_lateral_down_count"] /= 15  # Down max
    
    print(*detection_data.items(), sep='\n')

    # Result
    final_result_frontal = (
        detection_data["head_frontal_angle_mean"] * WEIGHTS["frontal_angle_mean_weight"]
        + detection_data["total_frontal_down_time"] * WEIGHTS["frontal_down_time_weight"]
        + detection_data["total_frontal_down_count"] * WEIGHTS["frontal_down_count_weight"]
    )

    final_result_lateral = (
        detection_data["head_lateral_angle_mean"] * WEIGHTS["lateral_angle_mean_weight"]
        + detection_data["total_lateral_down_time"] * WEIGHTS["lateral_down_time_weight"]
        + detection_data["total_lateral_down_count"] * WEIGHTS["lateral_down_count_weight"]
    )

    landmark = (
        final_result_frontal * WEIGHTS["frontal_weight"]
        + final_result_lateral * WEIGHTS["lateral_weight"]
    ) / 2

    return DetectionData(round(landmark, 1), detection_data)


if __name__ == "__main__":
    # Open the video file
    cap = cv.VideoCapture(r"drowsiness\testing\tired_0.mp4")
    fps = cap.get(cv.CAP_PROP_FPS)
    interval = int(round(fps / 10))
    count = 0
    frame_number = 0
    prefix = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if count % interval == 0:
                frame_number += 1
                if frame_number < 10:
                    cv.imwrite(
                        r"drowsiness\testing\frames\frame_00%d.png" % frame_number,
                        frame,
                    )
                elif 10 <= frame_number < 100:
                    cv.imwrite(
                        r"drowsiness\testing\frames\frame_0%d.png" % frame_number, frame
                    )
                else:
                    cv.imwrite(
                        r"drowsiness\testing\frames\frame_%d.png" % frame_number, frame
                    )

            count += 1
        else:
            break

    cap.release()
    classifier = KSSClassifier(0, 0, 0)

    video = create_frame_list()
    mp_results = []

    for frame in video:
        mp_results.append(MediapipeDetector["images"].process(frame))

    head_result = execute(mp_results, video[0].shape)

    classifier.set_results(None, head_result, None)

    metrics = list(head_result.data.keys())
    values = list(head_result.data.values())

    fig = plt.figure(figsize=(13, 8))
    plt.bar(metrics, values, color="g", width=0.4)
    plt.xlabel("Metrics", fontsize=10)
    plt.xticks(rotation=15, ha="right")
    plt.ylabel("Values")
    plt.title(f"Result {head_result.result}")
    plt.show()
