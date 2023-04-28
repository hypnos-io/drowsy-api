import asyncio

import socketio

from ws.client import SocketManager
from detection.drowsiness import Drowsy


sio_client = socketio.AsyncClient()
fatigue_detector = Drowsy()

server = SocketManager(sio_client, fatigue_detector)

asyncio.wait()