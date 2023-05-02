import numpy as np
import cv2 as cv

import base64

def ndarray_to_base64(image: np.ndarray, type: str = '.jpg') -> str:
    _, buffer = cv.imencode(type, image)
    
    bytes = np.array(buffer).tobytes()

    base_64 = base64.b64encode(bytes).decode('utf-8')

    return base_64

def base64_to_ndarray(base_64: str) -> np.ndarray: 
    _, data = base_64.split(',')

    # Decodifica a string dos dados em base64 para bytes
    bytes = base64.b64decode(data)

    # Converte os bytes em um numpy array
    byte_array = np.frombuffer(bytes, dtype=np.uint8)

    # Decodifica os bytes para uma imagem em preto e branco
    image = cv.imdecode(byte_array, cv.IMREAD_COLOR)

    return image

