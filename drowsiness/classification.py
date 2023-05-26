import numpy as np


from ws.entities import FatigueStatus
from drowsiness.detection.detector import DetectionData


class KSSClassifier:
    def __init__(self, eyes: DetectionData = DetectionData(0, {}), head: DetectionData = DetectionData(0, {}), mouth: DetectionData = DetectionData(0, {})):
        self.eyes_result = eyes
        self.head_result = head
        self.mouth_result = mouth

    def __calculate_kss_score(self):
        mouth = self.mouth_result.result
        eye = self.eyes_result.result
        head = self.head_result.result
        print('Head:', head)
        print('Eyes:', eye)
        print('Mouth:', mouth)

        """
        head:  [head], [eyes], [mouth]
        eyes:  [head], [eyes], [mouth]
        mouth: [head], [eyes], [mouth]
        """
        comparison_matrix = np.array([[1, 1 / 3, 5], [3, 1, 3], [1 / 5, 1 / 3, 1]])

        priority_vector = np.sum(comparison_matrix, axis=1) / np.sum(comparison_matrix)

        head_weight = priority_vector[0]
        eye_weight = priority_vector[1]
        mouth_weight = priority_vector[2]

        overall_tiredness = (
            (eye_weight * eye) + (mouth_weight * mouth) + (head_weight * head)
        )


        return int(round(overall_tiredness, 1) * 10)

    def classify(self):
        return self.__calculate_kss_score()

    def set_results(self, eye=None, head=None, mouth=None):
        if eye:
            self.eyes_result = eye

        if head:
            self.head_result = head

        if mouth:
            self.mouth_result = mouth

    def status(self) -> FatigueStatus:
        return {
            "kssScale": self.classify(),
            "detection": {
                "eyes": self.eyes_result.to_dict(),
                "head": self.head_result.to_dict(),
                "mouth": self.mouth_result.to_dict(),
            },
        }
