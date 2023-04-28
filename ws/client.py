from dataclasses import dataclass

import socketio

from . import connection_handlers, encoding
from ws.entities import FatigueStatus


@dataclass(frozen=True)
class HypnosData:
    id: str
    employeeId: str
    workstation: str
    fps: int

@dataclass(frozen=True)
class DrowsyRequest(HypnosData):
    images: list[str]

@dataclass(frozen=True)
class DrowsyResponse(HypnosData):
    imageStatus: FatigueStatus


class SocketManager():
    def __init__(self, client: socketio.AsyncClient, detector) -> None:
        self.client: socketio.AsyncClient = client
        self.detector = detector

        self._handle_connection(self.client)
        self._handle_processing(self.client)

    def _handle_connection(self, client: socketio.AsyncClient) -> None:
        client.on(connection_handlers.connect)
        client.on(connection_handlers.connect_error)
        client.on(connection_handlers.disconnect)

    def _handle_processing(self, client: socketio.AsyncClient) -> None:
        client.on('process-image', self._process)

    async def _process(self, data: DrowsyRequest) -> DrowsyResponse:
        images = [encoding.base64_to_ndarray(image) for image in data.images]

        response = await self.detector(images)

        await self.client.emit('notify-status', response)