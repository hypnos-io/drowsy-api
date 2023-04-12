from abc import ABC, abstractmethod


class AbstractDetector(ABC):
    @abstractmethod
    def execute(self, source):
        pass
