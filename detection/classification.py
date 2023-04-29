import sys
sys.path.append(r'drowsiness/')

import numpy as np
from detection.eye_insight_detector import EyeInsightDetector
from detection.mouth_detector import MouthDlibDetector
from detection.eye_mediapipe import create_frame_list
from detection.head_detector import HeadDetector

class KSSClassifier:
    def __init__(self, eyes_result, head_result, mouth_result):
        self.__eyes_result = eyes_result
        self.__head_result = head_result
        self.__mouth_result = mouth_result

    def __calculate_kss_score(self):
        mouth = self.__mouth_result.result
        eye = self.__eyes_result.result
        head = self.__head_result.result
        
        """
        head:  [head], [eyes], [mouth]
        eyes:  [head], [eyes], [mouth]
        mouth: [head], [eyes], [mouth]
        """
        comparison_matrix = np.array([[1, 3, 5], [1/3, 1, 3], [1/5, 1/3, 1]])

        priority_vector = np.sum(comparison_matrix, axis=1) / np.sum(comparison_matrix)

        head_weight = priority_vector[0]
        eye_weight = priority_vector[1]
        mouth_weight = priority_vector[2]

        overall_tiredness = (head_weight * head) + (eye_weight * eye) + (mouth_weight * mouth)
    
        #score = (mouth_score + eye_score + head_score) / 3
        score = overall_tiredness

        return int(round(score, 1) * 10)

    def classify(self):
        return self.__calculate_kss_score()

if __name__ == "__main__":
    frames = create_frame_list('test/tired', "png")
    mouth_detector = MouthDlibDetector()
    mouth_result = mouth_detector.execute(frames)
    print(mouth_result)
    eye_detector = EyeInsightDetector()
    head_detector = HeadDetector()
    eyes_result = eye_detector.execute(frames)
    head_result = head_detector.execute(frames)
    
    kss = KSSClassifier(eyes_result, head_result, mouth_result)
    
    print(kss.classify())
