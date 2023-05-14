from time import time

import cv2 as cv
import numpy as np

from detection import detector


OUTER_LIP = slice(48, 60)
INNER_LIP = slice(60, 68)


class MouthDlibDetector(detector.DlibDetector):
    def __init__(self, fps=24, min_area=200, min_duration=4) -> None:
        super().__init__()
        self._frame_rate = fps
        self._frame_length = 1 / fps

        self._yawn_area = min_area
        self._yawn_duration = min_duration

    def _handle_frame(self, frame):
        faces = self._detect_faces(frame)

        data = {
            "inner_area": self._yawn_area,
            "vertical_aperture": 0,
            "horizontal_aperture":0
        }

        for face in faces:
            landmarks = self._detect_landmarks(frame, face)
            landmarks = np.array(landmarks.parts())

            inner = np.array(
                [self.point_tuple(point) for point in landmarks[INNER_LIP]]
            )

            data["inner_area"] = cv.contourArea(inner)
            data["vertical_aperture"] = cv.norm(inner[2] - inner[-2])
            data["horizontal_aperture"] = cv.norm(inner[0] - inner[4])

        return data

    def execute(self, images) -> detector.DetectionData:
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

        maximum = np.max([data["inner_area"] for data in frame_data])
        area = np.array(
            [((data["inner_area"] - 0) / (maximum - 0)) for data in frame_data]
        )

        result = np.mean(area)
        detection_data["frames"] = frame_data

        return detector.DetectionData(result, detection_data)


if __name__ == "__main__":
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a camera")
        exit()

    mouth_detector = MouthDlibDetector()
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

        if time_elapsed > 1.0 / mouth_detector._frame_rate:
            prev = time()

            data = mouth_detector._handle_frame(frame)

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
