from typing import TypedDict

import socketio

from . import connection_handlers, encoding
from ws.entities import FatigueStatus


class DrowsyRequest(TypedDict, total=True):
    id: str
    employeeId: str
    workstation: str
    fps: int
    images: list[str]


class DrowsyResponse(TypedDict, total=True):
    id: str
    employeeId: str
    workstation: str
    fps: int
    imageStatus: FatigueStatus


class SocketManager:
    def __init__(self, client: socketio.Client, detector, url: str) -> None:
        self.client: socketio.AsyncClient = client
        self.detector = detector

        self._handle_connection(self.client, url)
        self._handle_processing(self.client)

        self.client.wait()

    def _handle_connection(self, client: socketio.Client, url) -> None:
        client.on(connection_handlers.connect)
        client.on(connection_handlers.connect_error)
        client.on(connection_handlers.disconnect)

        client.connect(url, wait_timeout=10)

    def _handle_processing(self, client: socketio.Client) -> None:
        client.on("process-image", self._process)

    def _process(self, data: DrowsyRequest) -> DrowsyResponse:
        images = [encoding.base64_to_ndarray(image) for image in data["images"]]

        imageStatus = self.detector(images)

        response: DrowsyResponse = {
            "id": data["id"],
            "employeeId": data["employeeId"],
            "workstation": data["workstation"],
            "fps": data["fps"],
            "imageStatus": imageStatus,
        }

        self.client.emit("notify-status", response)
