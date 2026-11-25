"""Pinned PEP 503 PyPI HTTP-over-Unix-socket proxy helpers shared by ``AgentSandbox``."""

from __future__ import annotations

import http.client
import ipaddress
import re
import socket
import ssl
from urllib.parse import unquote, urlsplit


def https_request(host, path, headers):
    """Open a verified HTTPS GET connection without redirects or non-global DNS targets."""
    answers = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    addresses = list(dict.fromkeys(answer[4][0] for answer in answers))
    if not addresses or any(
        not ipaddress.ip_address(address.split("%", 1)[0]).is_global
        for address in addresses
    ):
        raise OSError("package_host_dns_rejected")
    secure = ssl.create_default_context()
    last_error = None
    for address in addresses:
        raw = socket.socket(socket.AF_INET6 if ":" in address else socket.AF_INET, socket.SOCK_STREAM)
        tls = None
        connection = None
        try:
            raw.settimeout(20)
            raw.connect((address, 443))
            tls = secure.wrap_socket(raw, server_hostname=host)
            connection = http.client.HTTPSConnection(host, timeout=30)
            connection.sock = tls
            connection.request("GET", path, headers=headers)
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                raise OSError("package_host_redirect_rejected")
            return connection, response
        except (OSError, http.client.HTTPException) as error:
            last_error = error
            for resource in (connection, tls, raw):
                if resource is not None:
                    try:
                        resource.close()
                    except OSError:
                        pass
        except BaseException:
            for resource in (connection, tls, raw):
                if resource is not None:
                    try:
                        resource.close()
                    except OSError:
                        pass
            raise
    raise last_error or OSError("package_host_unreachable")


def package_reply(client, status, body=b"", headers=None):
    """Write a single HTTP/1.1 response onto ``client`` and forbid caching."""
    phrases = {
        200: "OK",
        400: "Bad Request",
        403: "Forbidden",
        404: "Not Found",
        405: "Method Not Allowed",
        502: "Bad Gateway",
    }
    lines = [
        f"HTTP/1.1 {status} {phrases.get(status, 'Error')}",
        "Connection: close",
        "Cache-Control: no-store",
    ]
    for key, value in (headers or {}).items():
        if key.lower() in {"content-type", "content-length", "etag", "last-modified", "cache-control"}:
            lines.append(f"{key}: {value}")
    if not any(line.lower().startswith("content-length:") for line in lines):
        lines.append(f"Content-Length: {len(body)}")
    client.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("ascii") + body)


def serve_package_connection(client, *, package_port, https_request_fn=None, reply_fn=None):
    """Serve only PEP 503 indexes and package objects from pinned PyPI hosts."""
    do_https = https_request_fn or https_request
    send_reply = reply_fn or package_reply
    client.settimeout(10)
    data = bytearray()
    while b"\r\n\r\n" not in data and len(data) <= 16384:
        part = client.recv(2048)
        if not part:
            return
        data.extend(part)
    if len(data) > 16384 or b"\r\n\r\n" not in data:
        return send_reply(client, 400)
    head, _, body = bytes(data).partition(b"\r\n\r\n")
    lines = head.decode("iso-8859-1").split("\r\n")
    request = lines[0].split()
    headers = {}
    for line in lines[1:]:
        if ":" not in line:
            return send_reply(client, 400)
        key, value = line.split(":", 1)
        headers[key.lower().strip()] = value.strip()
    if (
        len(request) != 3
        or request[0] != "GET"
        or request[2] not in ("HTTP/1.0", "HTTP/1.1")
        or headers.get("host", "").lower() not in {f"127.0.0.1:{package_port}", f"localhost:{package_port}"}
        or headers.get("transfer-encoding")
        or headers.get("content-length", "0") != "0"
        or body
    ):
        return send_reply(client, 405)
    parsed = urlsplit(request[1])
    if parsed.scheme or parsed.netloc or parsed.query:
        return send_reply(client, 403)
    encoded_path = parsed.path
    path = unquote(encoded_path)
    if (
        "\\" in path
        or any(part in (".", "..") for part in path.split("/"))
        or any(ord(char) < 0x21 or ord(char) == 0x7F for char in path)
    ):
        return send_reply(client, 403)
    try:
        if path.startswith("/simple/"):
            suffix = path[len("/simple/"):]
            if (
                not suffix.endswith("/")
                or suffix.count("/") != 1
                or not suffix[:-1]
                or not all(char.isalnum() or char in "-_." for char in suffix[:-1])
            ):
                return send_reply(client, 404)
            normalized = re.sub(r"[-_.]+", "-", suffix[:-1]).lower()
            upstream_path = "/simple/" + normalized + "/"
            accept = headers.get("accept", "text/html")[:512]
            request_headers = {"Accept": accept, "User-Agent": "ni-rua-package-index/1"}
            connection, response = do_https("pypi.org", upstream_path, request_headers)
            try:
                if response.status != 200:
                    return send_reply(client, 404 if response.status == 404 else 502)
                raw = bytearray()
                while chunk := response.read(65536):
                    raw.extend(chunk)
                    if len(raw) > 4 * 1024 * 1024:
                        return send_reply(client, 502)
                content_type = response.getheader("Content-Type", "text/html")
                if "html" not in content_type.lower() and "json" not in content_type.lower():
                    return send_reply(client, 502)
                rewritten = bytes(raw).replace(
                    b"https://files.pythonhosted.org/",
                    f"http://127.0.0.1:{package_port}/files/".encode("ascii"),
                )
                return send_reply(
                    client,
                    200,
                    rewritten,
                    {"Content-Type": content_type, "Cache-Control": "no-store"},
                )
            finally:
                connection.close()
        if path.startswith("/files/"):
            file_path = path[len("/files"):]
            if (
                not file_path.startswith("/packages/")
                or len(file_path) > 4096
                or not file_path.rsplit("/", 1)[-1].lower().endswith(
                    (".whl", ".whl.metadata", ".tar.gz", ".zip", ".metadata")
                )
            ):
                return send_reply(client, 404)
            connection, response = do_https(
                "files.pythonhosted.org",
                encoded_path[len("/files"):],
                {"Accept-Encoding": "identity", "User-Agent": "ni-rua-package-download/1"},
            )
            try:
                if response.status != 200:
                    return send_reply(client, 404 if response.status == 404 else 502)
                content_length = response.getheader("Content-Length")
                headers_out = {"Content-Type": response.getheader("Content-Type", "application/octet-stream")}
                chunked = content_length is None
                lines = [
                    "HTTP/1.1 200 OK",
                    "Connection: close",
                    "Cache-Control: no-store",
                    f"Content-Type: {headers_out['Content-Type']}",
                ]
                if chunked:
                    lines.append("Transfer-Encoding: chunked")
                else:
                    lines.append(f"Content-Length: {content_length}")
                client.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("ascii"))
                while chunk := response.read(65536):
                    if chunked:
                        client.sendall(f"{len(chunk):X}\r\n".encode("ascii") + chunk + b"\r\n")
                    else:
                        client.sendall(chunk)
                if chunked:
                    client.sendall(b"0\r\n\r\n")
            finally:
                connection.close()
            return
        return send_reply(client, 404)
    except (OSError, http.client.HTTPException, ssl.SSLError, UnicodeError):
        try:
            send_reply(client, 502)
        except OSError:
            pass
