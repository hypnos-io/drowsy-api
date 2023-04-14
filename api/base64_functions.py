import numpy as np
from numpy import ndarray
import cv2
import base64

def get_base64_code(base64: str) -> str:
    result = ''
    for i in range(22, len(base64)):
        result += base64[i]
    return result

def base64_2_cvimage(base64_data: str) -> ndarray:
    base64_code = get_base64_code(base64_data)
    # Decodifica a string em base64 para bytes
    img_bytes = base64.b64decode(base64_code)
    # Converte os bytes em uma imagem em formato numpy array
    nparr = np.frombuffer(img_bytes, np.uint8)
    # Lê a imagem usando o OpenCV
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img