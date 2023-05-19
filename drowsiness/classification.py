import numpy as np


from ws.entities import FatigueStatus
from detection.detection.eye import EyeInsightDetector, create_frame_list
from detection.detection.mouth import MouthInsightDetector
from detection.detection.head import HeadDetector
from detection.detection.detector import insight_app


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
        comparison_matrix = np.array([[1, 1/3, 5], 
                                      [3, 1, 3], 
                                      [1/5, 1/3, 1]])

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

    def status(self) -> FatigueStatus:
        return {
            "kssScale": self.classify(),
            "detection": {
                "eyes": self.eyes_result.to_dict(),
                "head": self.head_result.to_dict(),
                "mouth": self.mouth_result.to_dict()
            }
        }

if __name__ == "__main__":
    fps = 10
    video = create_frame_list()
    if len(video) <= 0:
        print("Lista vazia.")
    else:
        print("Running detection...")
        app = insight_app()
        eye = EyeInsightDetector(fps=fps, app=app)
        mouth = MouthInsightDetector(fps=fps, app=app)
        head = HeadDetector(fps=fps)
        classifier = KSSClassifier(0, 0, 0)
        
        eye_result = eye.execute(video)
        print("Deteccção dos olhos completa. Iniciando MouthDetector")
        mouth_result = mouth.execute(video)
        print('Detecção da boca completa. Iniciando HeadDetector')
        head_result = head.execute(video)
        print("Detecção da cabeça completa. Iniciando classficação.")
        
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
        elif 0.4 <= kss < 0.7:
            print("Kinda tired")
        else:
            print("Tired")
        print(f"\t eyes: {eye_result.result} | {eye_result.data}")
        print(f"\t head: {head_result.result}| {head_result.data}")
        print(f"\t mouth: {mouth_result.result} | {mouth_result.data}")
