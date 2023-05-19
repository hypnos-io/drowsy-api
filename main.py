import socketio


from ws.client import SocketManager
from drowsiness import drowsy

sio_client = socketio.Client()
fatigue_detector = drowsy.detect

server = SocketManager(sio_client, fatigue_detector, "http://localhost:3000")