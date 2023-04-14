import asyncio
import socketio
from time import sleep
from connectionSocket.events.connection import on_connection, disconnect, connect_error
from connectionSocket.events.fatigue import receive_images

sio = socketio.AsyncClient()

sio.on('connect', lambda: on_connection())
sio.on('disconnect', lambda: disconnect())
sio.on('connect_error', lambda data: connect_error(data))
sio.on('process-image', lambda data: asyncio.ensure_future(receive_images(sio, data)))


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
