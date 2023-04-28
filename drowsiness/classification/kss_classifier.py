import sys
sys.path.append(r'drowsiness/')

from detection.eye_insight_detector import EyeInsightDetector
from detection.mouth_detector import MouthDlibDetector
from detection.eye_mediapipe import create_frame_list

class KSSClassifier:
    def __init__(self, eyes_result, head_result, mouth_result):
        self.__eyes_result = eyes_result
        #self.__head_result = head_result
        self.__mouth_result = mouth_result

    def __calculate_kss_score(self, weight_dictionary):
        mouth = self.__mouth_result.data
        eye = self.__eyes_result.data
        #head = self.__head_result.data
        
        mouth_score = (
              (mouth["yawn_time"] * weight_dictionary["yawn_time_weight"])
            + (mouth["yawn_percentage"] * weight_dictionary["yawn_percentage"])
        )
        
        eye_score = (
                  (eye["eye_opening"] * weight_dictionary["eye_opening_weight"])
                + eye["closed_eyes_time"] * weight_dictionary["close_eyes_time_weight"]
                + eye["blink_count"] * weight_dictionary["blink_count_weight"])
        print(mouth_score)
        print(eye_score)
        score = (mouth_score + eye_score) / 2

        return round(score, 1) * 10

    def classify(self, weight_dictionary):
        score = self.__calculate_kss_score(weight_dictionary)

        classification = f"[{score}] "

        if score <= 3:
            classification += (
                "Sem fadiga: Sentindo-se ativo, vital, alerta ou bem acordado"
            )
        elif score <= 6:
            classification += "Aviso: Funcionando em niveis elevados, mas nao no pico; capaz de se concentrar"
        else:
            classification += "Alerta: Sonolento, cansado; precisa de aviso sonoro"

        return classification

if __name__ == "__main__":
    frames = create_frame_list('test/tired', "png")
    eye_detector = EyeInsightDetector()
    mouth_detector = MouthDlibDetector()
    eyes_result = eye_detector.execute(frames)
    mouth_result = mouth_detector.execute(frames)
    head_result = None
    
    kss = KSSClassifier(eyes_result, head_result, mouth_result)
    weight_dictionary = {
        "eye_opening_weight": 0.4,"blink_count_weight": 0.2, "close_eyes_time_weight": 0.4,
        "yawn_time_weight": 0.4, "yawn_percentage": 0.6
    }
    print(kss.classify(weight_dictionary))