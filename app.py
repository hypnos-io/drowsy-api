from flask import Flask
import socketio

app = Flask(__name__)

sio = socketio.Client()
sio.connect('http://localhost:3333')

@sio.event
def connect():
    print('connected')


@sio.event
def connect_error(e):
    print(f'Error: {e}')


@sio.on('server1 server2')
def receive_images(images):
    pass

if __name__ == '__main__':
    sio.connect('http://localhost:3333')
    sio.wait()


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"