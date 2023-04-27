from dataclasses import dataclass

import socketio
import numpy as np

from ws import connection_handlers, encoding


@dataclass(frozen=True)
class SocketData:
    id: str
    employeeId: str
    workstation: str

@dataclass(frozen=True)
class DrowsyRequest(SocketData):
    images: np.ndarray

@dataclass(frozen=True)
class DrowsyResponse(SocketData):
    imageStatus: str


class SocketManager():
    def __init__(self, detector) -> None:
        self.client = socketio.AsyncClient()
        self.detector = detector

        self._handle_connection(self.client)
        self._handle_processing(self.client)

    def _handle_connection(self, client) -> None:
        client.on(connection_handlers.connect)
        client.on(connection_handlers.connect_error)
        client.on(connection_handlers.disconnect)

    def _handle_processing(self, client) -> None:
        client.on('process-image', self._process)

    async def _process(self, data: DrowsyRequest) -> DrowsyResponse:
        
        images = [encoding.base64_to_ndarray(image) for image in data.images]

        response = self.detector(images)

        await self.client.emit('notify-status', response)