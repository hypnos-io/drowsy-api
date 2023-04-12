from time import time
from os import path

import dlib
import cv2 as cv
import numpy as np

from drowsiness import DetectionData

MODULE_DIR = path.dirname(path.abspath(__file__))

PREDICTOR_FACE_68 = dlib.shape_predictor(
    path.join(MODULE_DIR, "models", "shape_predictor_68_face_landmarks.dat")
)
DETECTOR_FHOG = dlib.get_frontal_face_detector()


INNER_LIP = slice(60, 68)
OUTER_LIP = slice(48, 60)


def point_tuple(point):
    return (point.x, point.y)


class MouthDetector:
    def __init__(self, fps=24, min_area=200, min_duration=4) -> None:
        self._frame_rate = fps
        self._frame_length = 1 / fps

        self._yawn_area = min_area
        self._yawn_duration = min_duration

        self._detector = DETECTOR_FHOG
        self._predictor = PREDICTOR_FACE_68

    def _detect_faces(self, source):
        faces = self._detector(source)

        return faces

    def _detect_landmarks(self, source, face):
        landmarks = self._predictor(source, face)

        return landmarks

    def execute(self, images):
        detection_data = {"yawn_count": 0, "yawn_frame_count": 0}

        frame_data = []
        yawn_frames = 0
        for frame in images:
            data = self._handle_frame(frame)

            if data["inner_area"] > self._yawn_area:
                yawn_frames += 1
            elif yawn_frames > 0:
                detection_data["yawn_count"] += 1
                detection_data["yawn_frame_count"] += yawn_frames
                yawn_frames = 0

            frame_data.append(data)

        detection_data["yawn_percentage"] = detection_data["yawn_frame_count"] / len(
            images
        )
        detection_data["yawn_time"] = (
            detection_data["yawn_frame_count"] * self._frame_length
        )

        maximum = np.max(area)
        area = np.array(
            [(data["inner_area"] - 0) / (maximum - 0)] for data in frame_data
        )

        result = np.mean(area)

        return {
            "result": result,
            **detection_data,
            "frames": frame_data,
        }  # TO-DO Converter saída para drowsiness.DetectionData()

    def _handle_frame(self, image):
        faces = self._detect_faces(image)

        data = {}

        for face in faces:
            landmarks = self._detect_landmarks(image, face)
            landmarks = np.array(landmarks.parts())

            inner = np.array([point_tuple(point) for point in landmarks[INNER_LIP]])

            data["inner_area"] = cv.contourArea(inner)
            data["vertical_aperture"] = cv.norm(inner[2] - inner[-2])
            data["horizontal_aperture"] = cv.norm(inner[0] - inner[4])

        return data


if __name__ == "__main__":
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a camera")
        exit()

    mouth = MouthDetector()
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
        elif key == ord("r"):
            testing = not testing

        if time_elapsed > 1.0 / mouth._frame_rate:
            prev = time()

            data = mouth._handle_frame(frame)

            y = 20
            for key, value in data.items():
                cv.putText(
                    frame,
                    f"{key}: {int(value)}",
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
