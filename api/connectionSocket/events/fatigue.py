import time
from domain.entities.fatigue import FatigueStatus, example_status
from base64_functions import base64_2_cvimage
import sys
sys.path.append('C:/Users/callidus/Desktop/Repositórios Github/drowsy-api')
from drowsiness.drowsiness import DrowsinessDetection
from config import status

class SocketDataRequest:
    def __init__(self, id: str, employee_id: str, workstation: str, images: list[str]):
        self.id = id
        self.employee_id = employee_id
        self.workstation = workstation
        self.images = images

    def get_dictionary(self) -> dict:
        dictionary = {
            'id': self.id,
            'employeeId': self.employee_id,
            'workstation': self.workstation,
            'images': self.images
        }
        return dictionary


class SocketDataResponse:
    def __init__(self, id: str, employee_id: str, workstation: str, image_status: FatigueStatus):
        self.id = id
        self.employee_id = employee_id
        self.workstation = workstation
        self.image_status = image_status

    def get_dictionary(self):
        dictionary = { 
            'id': self.id,
            'employeeId': self.employee_id,
            'workstation': self.workstation,
            'imageStatus': self.image_status.get_dictionary(),
        }
        return dictionary


async def send_response(sio, response: SocketDataResponse):
    print(f'[WS] Enviando resposta de {response.employee_id} para Hypnos API')
    await sio.emit('notify-status', response.get_dictionary())

async def receive_images(sio, data):
    global status
    request = SocketDataRequest(id=data['id'], employee_id=data['employeeId'], workstation=data['workstation'], images=data['images'])
    bgr_image = base64_2_cvimage(request.images[0])

    if status == None:
        status = DrowsinessDetection()
   # print(status.detectDrowsiness(bgr_image, 10)) # o valor bgr do frame está sendo utilizado como parâmetro do método detectDrowsiness
                                              # e impresso

    if len(status.getFPF()) == 0:
        status.setFPF((time.perf_counter(), 0))

    else:
        status.setFPF((time.perf_counter(), time.perf_counter() - (status.getFPF()[len(status.getFPF()) - 1])[0]))

    # print(status.getFPF())

    response = SocketDataResponse(id=request.id, 
                                  employee_id=request.employee_id, 
                                  workstation=request.workstation, 
                                  image_status=example_status)

    await send_response(sio, response)