from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
socket = SocketIO(app)

@socket.on('connect')
def handle_connect():
    print('Novo cliente conectado')

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"