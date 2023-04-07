import json

class DetectionData():
    def __init__(self, result, dictionary):
        self.result = result
        self.dictionary = dictionary
        
    def to_json(self):
        new_dict = {"result": result}
        new_dict.update(self.dictionary)

        json_data = json.dumps(new_dict)
        
        return json_data