from time import time
import sys
#sys.path.append(r'C:/Users/Callidus/Documents/drowsy-api')


import cv2 as cv
import numpy as np

from detection.detector import DlibDetector
from drowsiness.classification.detection_data import DetectionData


LEFT_EYE = slice(36, 42)
RIGHT_EYE = slice(42, 48)


class EyeDlibDetector(DlibDetector):
    def __init__(self, blink_threshold, fps=10, ear_threshold=0.20):
        super().__init__()
        self._frame_rate = fps
        self._frame_length = 1 / fps

        self._ear_treshhold = ear_threshold
        self._blink_threshold = blink_threshold

    def _calculate_ear(self, eye):
        vertical_distl = np.linalg.norm(eye[1] - eye[-1])
        vertical_distr = np.linalg.norm(eye[2] - eye[-2])
        horizontal_dist = np.linalg.norm(eye[0] - eye[3])

        eye_aspect_ratio = (vertical_distl + vertical_distr) / (2.0 * horizontal_dist)

        return eye_aspect_ratio

    def _handle_frame(self, frame):
        faces = self._detect_faces(frame)

        data = {}

        for face in faces:
            landmarks = self._detect_landmarks(frame, face)
            landmarks = np.array(landmarks.parts())

            left_eye = np.array(
                [self.point_tuple(point) for point in landmarks[LEFT_EYE]]
            )
            right_eye = np.array(
                [self.point_tuple(point) for point in landmarks[RIGHT_EYE]]
            )

            left_ear = self._calculate_ear(left_eye)
            right_ear = self._calculate_ear(right_eye)

            data["ear"] = np.mean((left_ear, right_ear))

        return data

    def execute(self, images):
        detection_data = {"blink_count": 0, "closed_frame_count": 0}

        frame_data = []
        blink_frames = 0
        for frame in images:
            data = self._handle_frame(frame)

            if data["ear"] < self._ear_treshhold:
                blink_frames += 1
                detection_data["closed_frame_count"] += 1
            else:
                if blink_frames >= self._blink_threshold:
                    detection_data["blink_count"] += 1
                blink_frames = 0

            frame_data.append(data)

        ear = np.mean(data["ear"] for data in frame_data)
        detection_data["closed_time"] = (
            detection_data["closed_frame_count"] * self._frame_length
        )

        result = ear
        return DetectionData(result, detection_data)


if __name__ == "__main__":
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a camera")
        exit()

    detector = EyeDlibDetector(1)
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

        if time_elapsed > 1.0 / detector._frame_rate:
            prev = time()

            data = detector._handle_frame(frame)

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
