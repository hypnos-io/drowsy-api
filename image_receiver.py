import asyncio
import socketio
from time import sleep
import random


class SocketDataRequest:
    def __init__(self, id: str, images: list[str]):
        self.id = id
        self.images = images


class ImageStatus:
    def __init__(self, kss: int, detection: dict[str, str | int]):
        self.kss = kss
        self.detection = detection

    def get_dictionary(self):
        dictionary = {
            "kss": self.kss,
            "detection": self.detection
        }
        return dictionary


class SocketDataResponse:
    def __init__(self, id: str, image_status: ImageStatus):
        self.id = id
        self.image_status = image_status

    def get_dictionary(self):
        dictionary = { 
            "id": self.id,
            "imageStatus": self.image_status.get_dictionary()
        }
        return dictionary


# dictionairy with the possible status returned to hypnos api after analysis
status = {"1": 'Neutro', "2": "Há fadiga", "3": "Não há fadiga", "4": "Usuário não detectado"}

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
    
    
    '''
    Enviar imagens para análise
    '''

    await sio.emit('notify-status', random.choice(list(status.values())))


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
