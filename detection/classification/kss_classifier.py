import numpy as np

import numpy as np

class KSSClassifier:
    def __init__(self, num_frames, eyes_result, head_result):
        self.num_frames = num_frames
        self.eyes_result = eyes_result
        self.head_result = head_result
    
    def calculate_eye_closed_percentage(self, eye_closed_frames):
        return eye_closed_frames / self.num_frames
    
    def calculate_kss_score(self, weight_dictionary):
        #eye_closed_percentage = self.calculate_eye_closed_percentage(eye_closed_frames)
        # Pesos para características diferentes
        eye_closed_weight = weight_dictionary["closed eye weight"]
        blink_weight = weight_dictionary["blink weight"]
        ear_eye_weight = weight_dictionary["eye ear weight"]
        # head_tilt_weight = weight_dictionary["head tilt weight"]
        
        # Normaliza os resultados
        num_blinks_norm = ( self.eyes_result["blinks"] / (20/60 * self.num_frames)) * 60 / 20
        average_ear_norm = self.eyes_result["eye_opening"] / 0.4
        closed_eyes_norm = self.eyes_result["closed_eyes"] / (self.num_frames / 10)
        
        
        score = (eye_closed_weight * closed_eyes_norm) + (blink_weight * num_blinks_norm)\
            + (ear_eye_weight * average_ear_norm)
        
        scores = {
            "blink": num_blinks_norm,
            "ear": average_ear_norm,
            "closed": closed_eyes_norm,
            "score": score
        }
        print(f"KSS SCORE: {scores}")
        return round(score, 4) * 10
    
    def classify(self, weight_dictionary):
        score = self.calculate_kss_score(weight_dictionary)
        
        if score <= 3:
            return f"[{score}] Sem fadiga: Sentindo-se ativo, vital, alerta ou bem acordado"
        elif score <= 6:
            return f"[{score}] Aviso: Funcionando em niveis elevados, mas nao no pico; capaz de se concentrar"
        else:
            return f"[{score}] Alerta: Sonolento, cansado; precisa de aviso sonoro" 
