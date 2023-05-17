import sys

import numpy as np
from detection.detection.eye_insight_detector import EyeInsightDetector, create_frame_list
from detection.detection.mouth_insight_detector import MouthInsightDetector 
from detection.detection.head_detector import HeadDetector

# from detection.eye_insight_detector import EyeInsightDetector, create_frame_list
# from detection.mouth_insight_detector import MouthInsightDetector
# from detection.head_detector import HeadDetector


class KSSClassifier:
    def __init__(self, eyes_result, head_result, mouth_result):
        self.eyes_result = eyes_result
        self.head_result = head_result
        self.mouth_result = mouth_result

    def __calculate_kss_score(self):
        mouth = self.mouth_result.result
        eye = self.eyes_result.result
        head = self.head_result.result
        
        """
        head:  [head], [eyes], [mouth]
        eyes:  [head], [eyes], [mouth]
        mouth: [head], [eyes], [mouth]
        """
        comparison_matrix = np.array([[1, 3, 5], 
                                      [1/3, 1, 1/3], 
                                      [1/5, 3, 1]])

        priority_vector = np.sum(comparison_matrix, axis=1) / np.sum(comparison_matrix)

        head_weight = priority_vector[0]
        eye_weight = priority_vector[1]
        mouth_weight = priority_vector[2]

        overall_tiredness = (head_weight * head) + (eye_weight * eye) + (mouth_weight * mouth)
    
        score = overall_tiredness

        return int(round(score, 1) * 10)

    def classify(self):
        return self.__calculate_kss_score()
    
    def set_results(self, eyes_result = None, head_result = None, mouth_result = None):
        if eyes_result:
            self.eyes_result = eyes_result

        if head_result:
            self.head_result = head_result
            
        if mouth_result:
            self.mouth_result = mouth_result

if __name__ == "__main__":
    fps = 10
    video = create_frame_list()
    if len(video) <= 0:
        print("Lista vazia.")
    else:
        eye = EyeInsightDetector(fps=fps)
        mouth = MouthInsightDetector(fps=fps)
        head = HeadDetector(fps=fps)
        classifier = KSSClassifier(0, 0, 0)
        
        eye_result = eye.execute(video)
        mouth_result = mouth.execute(video)
        head_result = head.execute(video)
        
        classifier.set_results(
                eye_result,
                head_result,
                mouth_result
            )

        kss = classifier.classify()
        print("============ RESULTADO ===========")
        print(f"KSS: {kss}")
        if kss < 0.4:
            print("Not tired")
        elif 0.4 <= response.result < 0.7:
            print("Kinda tired")
        else:
            print("Tired")
        print(f"\t eyes: {eye_result.result} | {eye_result.data}")
        print(f"\t head: {head_result.result}| {head_result.data}")
        print(f"\t mouth: {mouth_result.result} | {mouth_result.data}")
