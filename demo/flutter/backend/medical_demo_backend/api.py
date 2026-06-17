"""Pure-Python WSGI API for the medical demo backend."""

from __future__ import annotations

import base64
import json
import os
from email.parser import BytesParser
from email.policy import default
from typing import Callable, Iterable, Tuple
from wsgiref.simple_server import make_server

from .service import build_default_service
from .types import ChatMessage, ChatRequest, ConsultationRequest


StatusHeadersBody = Tuple[str, list[tuple[str, str]], bytes]


class MedicalDemoApp:
    """Simple WSGI router."""

    def __init__(self, service=None) -> None:
        self.service = service or build_default_service()

    def __call__(self, environ: dict, start_response: Callable) -> Iterable[bytes]:
        status, headers, body = self.handle(environ)
        start_response(status, headers)
        return [body]

    def handle(self, environ: dict) -> StatusHeadersBody:
        method = environ.get("REQUEST_METHOD", "GET").upper()
        path = environ.get("PATH_INFO", "/")
        if method == "OPTIONS":
            return self._empty_response("204 No Content")
        if method == "GET" and path == "/":
            return self._html_response("200 OK", self._root_page())
        if method == "GET" and path == "/health":
            return self._json_response("200 OK", self.service.health())
        if method == "POST" and path == "/v1/consult":
            try:
                payload = self._parse_request(environ)
                request = ConsultationRequest(
                    patient_text=str(payload.get("patient_text", "")),
                    image_bytes=payload.get("image_bytes"),
                    image_filename=str(payload.get("image_filename", "")),
                    client_request_id=str(payload.get("client_request_id", "")),
                    device_platform=str(payload.get("device_platform", "")),
                )
                response = self.service.consult(request)
                return self._json_response("200 OK", response.to_dict())
            except ValueError as exc:
                return self._json_response("400 Bad Request", {"error": str(exc)})
        if method == "POST" and path == "/v1/chat":
            try:
                payload = self._parse_request(environ)
                request = ChatRequest(
                    messages=[
                        ChatMessage(
                            role=str(item.get("role", "user")),
                            content=str(item.get("content", "")),
                        )
                        for item in payload.get("messages", [])
                        if isinstance(item, dict)
                    ],
                    image_bytes=payload.get("image_bytes"),
                    image_filename=str(payload.get("image_filename", "")),
                    client_request_id=str(payload.get("client_request_id", "")),
                    device_platform=str(payload.get("device_platform", "")),
                    response_mode=str(payload.get("response_mode", "live_gpu")),
                )
                response = self.service.chat(request)
                return self._json_response("200 OK", response.to_dict())
            except ValueError as exc:
                return self._json_response("400 Bad Request", {"error": str(exc)})
        return self._json_response("404 Not Found", {"error": "not_found"})

    @staticmethod
    def _root_page() -> str:
        return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Medical Demo Backend</title>
  <style>
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #172033; background: #f4f7fb; }
    main { max-width: 880px; margin: 56px auto; padding: 0 24px; }
    .card { background: white; border: 1px solid #d2dce9; border-radius: 10px; padding: 28px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08); }
    h1 { margin: 0 0 8px; font-size: 32px; }
    p { line-height: 1.55; color: #526078; }
    a { color: #13797d; font-weight: 650; }
    code { background: #eaf1fb; border: 1px solid #d2dce9; border-radius: 5px; padding: 2px 6px; }
    ul { line-height: 1.9; }
    .note { margin-top: 20px; padding: 14px 16px; background: #fff7e8; border-left: 4px solid #9a5a00; color: #6f3f00; }
  </style>
</head>
<body>
  <main>
    <section class="card">
      <h1>Medical Demo Backend</h1>
      <p>This port is the API backend, not the Flutter web app. Use the endpoints below.</p>
      <ul>
        <li>Backend health: <a href="/health"><code>/health</code></a></li>
        <li>Chat API: <code>POST /v1/chat</code></li>
        <li>Consult API: <code>POST /v1/consult</code></li>
        <li>Flutter UI: <a href="http://127.0.0.1:8021">http://127.0.0.1:8021</a></li>
        <li>Real-case evidence view: <a href="http://127.0.0.1:8022/REAL_CASES_AUDIENCE_VIEW.html">http://127.0.0.1:8022/REAL_CASES_AUDIENCE_VIEW.html</a></li>
      </ul>
      <div class="note">Research decision support demo only. This is not clinical deployment and not a diagnosis-from-image system.</div>
    </section>
  </main>
</body>
</html>
"""

    def _parse_request(self, environ: dict) -> dict:
        content_type = environ.get("CONTENT_TYPE", "")
        if content_type.startswith("application/json"):
            length = int(environ.get("CONTENT_LENGTH") or "0")
            body = environ["wsgi.input"].read(length) if length else b"{}"
            payload = json.loads(body.decode("utf-8"))
            image_bytes = None
            if payload.get("image_base64"):
                image_bytes = base64.b64decode(str(payload["image_base64"]))
            return {
                "patient_text": str(payload.get("patient_text", "")),
                "messages": payload.get("messages") or [],
                "image_bytes": image_bytes,
                "image_filename": str(payload.get("image_filename", "")),
                "client_request_id": str(payload.get("client_request_id", "")),
                "device_platform": str(payload.get("device_platform", "")),
                "response_mode": str(payload.get("response_mode", "live_gpu")),
            }
        if content_type.startswith("multipart/form-data"):
            length = int(environ.get("CONTENT_LENGTH") or "0")
            body = environ["wsgi.input"].read(length) if length else b""
            parser = BytesParser(policy=default)
            message = parser.parsebytes(
                (
                    f"Content-Type: {content_type}\r\n"
                    "MIME-Version: 1.0\r\n\r\n"
                ).encode("utf-8")
                + body
            )
            fields = {}
            image_bytes = None
            image_filename = ""
            for part in message.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if not name:
                    continue
                filename = part.get_filename()
                payload = part.get_payload(decode=True)
                if filename:
                    if name == "image":
                        image_bytes = payload
                        image_filename = filename
                    continue
                fields[name] = (payload or b"").decode(part.get_content_charset() or "utf-8")
            return {
                "patient_text": fields.get("patient_text", ""),
                "messages": json.loads(fields.get("messages", "[]")),
                "image_bytes": image_bytes,
                "image_filename": image_filename,
                "client_request_id": fields.get("client_request_id", ""),
                "device_platform": fields.get("device_platform", ""),
                "response_mode": fields.get("response_mode", "live_gpu"),
            }
        raise ValueError("Unsupported content type. Use application/json or multipart/form-data.")

    @staticmethod
    def _cors_headers() -> list[tuple[str, str]]:
        return [
            ("Access-Control-Allow-Origin", "*"),
            ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
            ("Access-Control-Allow-Headers", "Content-Type"),
        ]

    @staticmethod
    def _empty_response(status: str) -> StatusHeadersBody:
        headers = [
            ("Content-Length", "0"),
            *MedicalDemoApp._cors_headers(),
        ]
        return status, headers, b""

    @staticmethod
    def _json_response(status: str, payload: dict) -> StatusHeadersBody:
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        headers = [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
            *MedicalDemoApp._cors_headers(),
        ]
        return status, headers, body

    @staticmethod
    def _html_response(status: str, html: str) -> StatusHeadersBody:
        body = html.encode("utf-8")
        headers = [
            ("Content-Type", "text/html; charset=utf-8"),
            ("Content-Length", str(len(body))),
            *MedicalDemoApp._cors_headers(),
        ]
        return status, headers, body


def create_app(service=None) -> MedicalDemoApp:
    """Create a new app instance."""
    return MedicalDemoApp(service=service)


def main() -> None:
    """Run a local development server."""
    app = create_app()
    host = os.getenv("MEDICAL_DEMO_HOST", "0.0.0.0")
    port = int(os.getenv("MEDICAL_DEMO_PORT", "8010"))
    with make_server(host, port, app) as httpd:
        print(f"medical-demo-backend listening on http://{host}:{port}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
