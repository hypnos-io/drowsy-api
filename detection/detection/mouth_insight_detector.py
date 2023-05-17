from time import time

#from detection.detection import detector
import cv2 as cv
import numpy as np
from detection import detector

OUTER_LIP = np.array([52, 64, 63, 67, 68, 61, 58, 59, 53, 56, 55])
INNER_LIP = np.array([65, 66, 62, 70, 69, 57, 60, 54])


class MouthInsightDetector(detector.InsightDetector):
    def __init__(self, fps=10, video_lenght=30, min_area=300, min_duration=4) -> None:
        super().__init__()

        self._yawn_area = min_area
        self._fps = fps
        self._yawn_duration = min_duration
        self.__video_length = video_lenght
        self._frame_length = 1 / fps
        self._max_num_yawn = 10


    def _handle_frame(self, frame):
        faces = self._detect_faces(frame)

        data = {
            "inner_area": self._yawn_area,
            "vertical_aperture": 0,
            "horizontal_aperture":0
        }

        for face in faces:
            landmarks = self._detect_landmarks(face)
            
            inner = np.array(landmarks[INNER_LIP])

            data["inner_area"] = cv.contourArea(inner)
            #data["vertical_aperture"] = cv.norm(inner[2] - inner[-2])
            #data["horizontal_aperture"] = cv.norm(inner[0] - inner[4])

        return data["inner_area"]

    def execute(self, images):
        detection_data = {"yawn_count": 0, "yawn_frame_count": 0}
        weights = {"yawn_count_weight": 0.2, "yawn_percentage_weight": 0.4, "yawn_time_weight": 0.4}

        area_array = []
        yawn_frames = 0
        for frame in images:
            inner_area = self._handle_frame(frame)

            if inner_area > self._yawn_area:
                yawn_frames += 1
            else:
                if yawn_frames > self._yawn_duration:
                    detection_data["yawn_count"] += 1
                    detection_data["yawn_frame_count"] += yawn_frames
                yawn_frames = 0

            area_array.append(inner_area)
            
        detection_data["yawn_count"] /= self._max_num_yawn
        detection_data["yawn_percentage"] = detection_data["yawn_frame_count"] / len(images)
        detection_data["yawn_time"] = (
            (detection_data["yawn_frame_count"] * self._frame_length) / self.__video_length
        )
        


        # area_array = (area_array - np.min(area_array)) / (np.max(area_array) - np.min(area_array))
        # area = np.mean(area_array)
        # result = np.mean(area)
        #detection_data["frames"] = area_array
        result = (
              (detection_data["yawn_count"] * weights["yawn_count_weight"])
            + (detection_data["yawn_percentage"] * weights["yawn_percentage_weight"])
            + (detection_data["yawn_time"] * weights["yawn_time_weight"])
        )

        return detector.DetectionData(round(result, 1), detection_data)


if __name__ == "__main__":
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a camera")
        exit()

    detector = MouthInsightDetector()
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

        if time_elapsed > 1.0 / detector._fps:
            prev = time()

            data = detector._handle_frame(frame)

            y = 20
            cv.putText(
                    frame,
                    f"inner_area: {data}",
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
