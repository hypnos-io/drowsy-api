import socketio


from ws.client import SocketManager
from detection.drowsiness import Drowsy

sio_client = socketio.Client()
fatigue_detector = Drowsy()

server = SocketManager(sio_client, fatigue_detector, "http://localhost:3000")