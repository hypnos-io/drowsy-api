from enum import Enum

class KssScaleEnum(Enum):
    EXTREMELY_ALERT = 1
    VERY_ALERT = 2
    ALERT = 3
    FAIRLY_ALERT = 4
    NEITHER_ALERT_NOR_SLEEPY = 5
    SOME_SIGNS_OF_SLEEPINESS = 6
    SLEEPY_BUT_NOT_EFFORT_TO_KEEP_ALERT = 7
    SLEEPY_SOME_EFFORT_TO_KEEP_ALERT = 8
    VERY_SLEEPY = 9


class FatigueDetectionInfo:
    def __init__(self, mouth: dict, eyes: dict, head: dict):
        self.mouth = mouth
        self.eyes = eyes
        self.head = head

    def get_dictionary(self):
        dictionary = {
            'mouth': self.mouth,
            'eyes': self.eyes,
            'head': self.eyes
        }
        return dictionary


class FatigueStatus:
    def __init__(self, kssScale: KssScaleEnum, detection: FatigueDetectionInfo):
        self.kssScale = kssScale
        self.detection = detection

    def get_dictionary(self):
        dictionary = {
            'kss': self.kssScale,
            'detection': self.detection.get_dictionary()
        }
        return dictionary