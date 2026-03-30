from __future__ import annotations

from typing import Dict

from fastapi import WebSocket


class ConnectionManager:
    """Хранит активные WebSocket-сессии и позволяет отправлять сообщения в конкретную сессию."""

    def __init__(self) -> None:
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str) -> None:
        self.active_connections.pop(session_id, None)

    async def send_message(self, session_id: str, message: str) -> None:
        ws = self.active_connections.get(session_id)
        if not ws:
            return
        await ws.send_text(message)


manager = ConnectionManager()
