import json


class DrowsinessDetection():

    def __init__(self) -> None:
        # self._eye = 
        # self._mouth = 
        # self._head = 
        ...

    def detectDrowsiness():
        ...



class DetectionData():

    def __init__(self, result, data) -> None:
        assert 0 <= result <= 1, "Detection result should be inbetween 0 and 1"

        self.result = result
        self.data = data

    def json(self) -> str:
        return json.dumps({
            'result': self.result,
            **self.data
        })