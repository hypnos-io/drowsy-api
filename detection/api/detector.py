from abc import ABC, abstractmethod

class Detector(ABC):
    @abstractmethod
    def cropROI(self, source):
        pass
    
    @abstractmethod
    def execute(self, source):
        pass