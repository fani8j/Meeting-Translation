from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

MessageHandler = Callable[[dict[str, Any]], Awaitable[None]]


class JsonlServer:
    def __init__(self, socket_path: Path, on_message: MessageHandler) -> None:
        self.socket_path = socket_path
        self.on_message = on_message
        self._server: asyncio.AbstractServer | None = None
        self._clients: set[asyncio.StreamWriter] = set()
        self._broadcast_lock = asyncio.Lock()

    async def start(self) -> None:
        if self.socket_path.exists():
            self.socket_path.unlink()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._server = await asyncio.start_unix_server(self._handle_client, path=self.socket_path)

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        clients = tuple(self._clients)
        self._clients.clear()
        for writer in clients:
            writer.close()
        await asyncio.gather(*(writer.wait_closed() for writer in clients), return_exceptions=True)
        if self.socket_path.exists():
            self.socket_path.unlink()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._clients.add(writer)
        try:
            while line := await reader.readline():
                try:
                    message = json.loads(line)
                    if not isinstance(message, dict):
                        raise ValueError("Message must be a JSON object")
                    await self.on_message(message)
                except (json.JSONDecodeError, ValueError) as error:
                    await self._write(writer, {"event": "protocol_error", "message": str(error)})
        finally:
            self._clients.discard(writer)
            writer.close()
            with contextlib.suppress(ConnectionError, BrokenPipeError):
                await writer.wait_closed()

    async def broadcast(self, payload: dict[str, Any]) -> None:
        async with self._broadcast_lock:
            failed: list[asyncio.StreamWriter] = []
            for writer in tuple(self._clients):
                try:
                    await self._write(writer, payload)
                except (ConnectionError, BrokenPipeError):
                    failed.append(writer)
            for writer in failed:
                self._clients.discard(writer)
                writer.close()

    @staticmethod
    async def _write(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
        writer.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
        await writer.drain()


async def send_command(socket_path: Path, command: str, **values: Any) -> None:
    reader, writer = await asyncio.open_unix_connection(socket_path)
    del reader
    writer.write(json.dumps({"command": command, **values}).encode("utf-8") + b"\n")
    await writer.drain()
    writer.close()
    await writer.wait_closed()
