import numpy as np


from ws.entities import FatigueStatus



class KSSClassifier:
    def __init__(self, eyes_result, head_result, mouth_result):
        self.eyes_result = eyes_result
        #self.head_result = head_result
        self.mouth_result = mouth_result

    def __calculate_kss_score(self):
        mouth = self.mouth_result.result
        eye = self.eyes_result.result
        #head = self.head_result.result
        #print(f"{mouth} | {eye} | {head}")

        """
        head:  [head], [eyes], [mouth]
        eyes:  [head], [eyes], [mouth]
        mouth: [head], [eyes], [mouth]
        """
        comparison_matrix = np.array([[1, 1 / 3, 1/5], [3, 1, 4], [5, 1 / 4, 1]])

        priority_vector = np.sum(comparison_matrix, axis=1) / np.sum(comparison_matrix)

        #head_weight = priority_vector[0]
        eye_weight = priority_vector[1]
        mouth_weight = priority_vector[2]

        overall_tiredness = (
            (eye_weight * eye) + (mouth_weight * mouth)
        )

        return int(round(overall_tiredness, 1) * 10)

    def classify(self):
        return self.__calculate_kss_score()

    def set_results(self, eyes_result=None, head_result=None, mouth_result=None):
        print(f"RESULTS: {eyes_result} | {head_result} | {mouth_result}")
        if eyes_result:
            self.eyes_result = eyes_result

        if head_result:
            self.head_result = head_result

        if mouth_result:
            self.mouth_result = mouth_result

    def status(self):
        return {
            "kssScale": self.classify(),
            "detection": {
                "eyes": self.eyes_result.to_dict(),
                #"head": self.head_result.to_dict(),
                "mouth": self.mouth_result.to_dict(),
            },
        }
    


