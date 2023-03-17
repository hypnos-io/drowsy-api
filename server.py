import asyncio
import socketio
from time import sleep
import random

sinais_fadiga = ['Funcionário com fadiga', 'Funcionário não apresentou fadiga']

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


@sio.on('server1 server2')
async def receive_images(image):
    print(image)

    '''
    Enviar imagens para análise
    '''

    # Envia uma resposta após análise da imagem
    sio.emit('drowsy_response', random.choice(sinais_fadiga))


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