import numpy as np


from ws.entities import FatigueStatus


class KSSClassifier:
    def __init__(self, eyes_result, head_result, mouth_result):
        self.eyes_result = eyes_result
        self.head_result = head_result
        self.mouth_result = mouth_result

    def __calculate_kss_score(self):
        mouth = self.mouth_result.result
        eye = self.eyes_result.result
        head = 0

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
            (head_weight * head) + (eye_weight * eye) + (mouth_weight * mouth)
        )

        score = overall_tiredness

        return int(round(score, 1) * 10)

    def classify(self):
        return self.__calculate_kss_score()

    def set_results(self, eyes_result=None, head_result=None, mouth_result=None):
        if eyes_result:
            self.eyes_result = eyes_result

        if head_result:
            self.head_result = head_result

        if mouth_result:
            self.mouth_result = mouth_result

    def status(self) -> FatigueStatus:
        return {
            "kssScale": self.classify(),
            "detection": {
                "eyes": self.eyes_result.to_dict(),
                # "head": self.head_result.to_dict(),
                "mouth": self.mouth_result.to_dict(),
            },
        }
