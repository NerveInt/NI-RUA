"""WebSocket-to-JSONL stdio bridge for the private Codex app-server Unix domain socket."""

from __future__ import annotations

import asyncio
import sys


def build_bridge_command(socket_path) -> tuple[str, ...]:
    """Return the SDK ``launch_args_override`` argv that bridges SDK stdio to ``socket_path``."""
    return (sys.executable, "-m", "ni_rua.cortex.agents.codex.bridge", "proxy", str(socket_path))


async def proxy(path: str) -> None:
    """Bridge JSONL stdin/stdout onto the pinned Unix WebSocket transport at ``path``."""
    import aiohttp

    reader = asyncio.StreamReader(limit=32 * 1024 * 1024)
    await asyncio.get_running_loop().connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin.buffer)
    async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=path)) as client:
        async with client.ws_connect("http://localhost/", max_msg_size=32 * 1024 * 1024) as ws:

            async def send():
                while line := await reader.readline():
                    await ws.send_str(line.decode().rstrip("\n"))
                await ws.close()

            async def receive():
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.TEXT:
                        sys.stdout.write(message.data + "\n")
                        sys.stdout.flush()

            tasks = [asyncio.create_task(send()), asyncio.create_task(receive())]
            try:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    socket_arg = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "proxy" else sys.argv[1]
    asyncio.run(proxy(socket_arg))
