import json

class DetectionData():
    def __init__(self, result, dictionary):
        self.result = result
        self.dictionary = dictionary
        
    def to_json(self):
        new_dict = {"result": result}
        self.dictionary.update(new_dict)

        json_data = json.dumps(self.dictionary)
        
        return json_data