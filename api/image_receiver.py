import asyncio
import socketio
from time import sleep
import random

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
@sio.on('server_to_drowsy')
async def receive_images(data):
    # getting the base64 url
    frame = list(data.values())[0]

    # printing images as base64 strings
    print(frame)
    # this algorithm gets one frame and tests if the image was really received
    #with open("imagensbase64.txt", "w") as base64:
       #  base64.write(str(frame))
    '''
    Enviar imagens para análise
    '''

    # A message is sent specifying the worker status based on their face
    await sio.emit('Status', random.choice(list(status.values())))


async def connect_to_server():
    while True:
        try:
            await sio.connect('http://localhost:3001')
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
