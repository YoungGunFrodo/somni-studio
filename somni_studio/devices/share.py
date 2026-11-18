"""Handing one sound file to a speaker on the network.

A cast device plays audio by fetching it over HTTP, but Somni Studio itself only listens on this
computer. So each message gets its own tiny server: one file, at a random address, answering only
the device it is meant for, for two minutes at most.
"""
from __future__ import annotations

import secrets
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import net

LIFETIME_S = 120
MAX_FETCHES = 4                  # cast devices fetch a file more than once (probe, then play)


def local_address_toward(device_ip: str) -> str:
    """This computer's address on the network the device is on."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((net.require_lan(device_ip), 9))      # no packet is sent
        return s.getsockname()[0]


def share_once(data: bytes, device_ip: str, content_type: str = "audio/wav") -> str:
    """Serve `data` to `device_ip` only; returns the URL. The server stops by itself."""
    host = local_address_toward(device_ip)
    path = f"/{secrets.token_urlsafe(16)}.wav"
    fetches = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server's naming
            if self.client_address[0] != device_ip or self.path != path or fetches["n"] >= MAX_FETCHES:
                self.send_error(404)
                return
            fetches["n"] += 1
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True, name="share-once").start()
    stop = threading.Timer(LIFETIME_S, server.shutdown)
    stop.daemon = True
    stop.start()
    return f"http://{host}:{server.server_address[1]}{path}"
