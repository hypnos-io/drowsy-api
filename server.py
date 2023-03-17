import asyncio
import socketio
from time import sleep

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


async def main():
    while True:
        try:
            await sio.connect('http://localhost:3001')
            break
        except socketio.exceptions.ConnectionError:
            sleep(5)

asyncio.run(main())