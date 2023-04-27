async def connect():
    print('[WS] Conectado no servidor.')

async def disconnect():
    print('[WS] Desconectado do servidor.')

async def connect_error(data):
    print('[WS] Não foi possível se conectar.')