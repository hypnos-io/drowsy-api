import asyncio
import socketio
from time import sleep
from base64_functions import base64_2_cvimage
from fatigue import FatigueStatus, example_status


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

sio = socketio.AsyncClient()

@sio.event
def connect():
    print('Conectado no servidor.')

@sio.event
def disconnect():
    print('Desconectado do servidor.')

@sio.event
def connect_error(data):
    print('Não foi possível se conectar.')


# this function is used to receive images from hypnos api and emit a response based on
# the worker facial status after analysis
@sio.on('process-image')
async def receive_images(data):
    request = SocketDataRequest(id=data['id'], employee_id=data['employeeId'], workstation=data['workstation'], images=data['images'])
    rgb_image = base64_2_cvimage(request.images[0])

    '''
    status = DrowsinessDetection().detectDrowsiness([rgb_image], fps)
    '''

    response = SocketDataResponse(id=request.id, 
                                  employee_id=request.employee_id, 
                                  workstation=request.workstation, 
                                  image_status=example_status).get_dictionary()

    await sio.emit('notify-status', response)


async def connect_to_server():
    while True:
        try:
            await sio.connect('http://localhost:3000')
            break
        except socketio.exceptions.ConnectionError:
            sleep(5)


async def main():
    await connect_to_server()

    while True:
        await asyncio.sleep(1)


if __name__ == '__main__':
    loop = asyncio.new_event_loop() 
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
    loop.close()
