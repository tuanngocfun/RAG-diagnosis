#!/usr/bin/env python3
"""HTTP server for Flutter web build that also reverse-proxies API requests
to the backend on port 8010.  This way the browser only needs one forwarded
port (8021) — no need to separately forward 8010."""

import http.server
import os
import sys
import urllib.request
import urllib.error


BACKEND_ORIGIN = os.getenv("MEDICAL_DEMO_BACKEND", "http://127.0.0.1:8010")

# Paths that should be proxied to the backend instead of served as files.
_PROXY_PREFIXES = ("/health", "/v1/")


class ProxyHandler(http.server.SimpleHTTPRequestHandler):
    """Serves static Flutter web files and proxies API paths to the backend."""

    def _should_proxy(self) -> bool:
        return any(self.path.startswith(p) for p in _PROXY_PREFIXES)

    # ---- static-file overrides (no-cache headers) ----

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    # ---- proxy helpers ----

    def _proxy(self, method: str = "GET", body: bytes | None = None):
        url = f"{BACKEND_ORIGIN}{self.path}"
        headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = resp.read()
                self.send_response(resp.status)
                for key in ("Content-Type", "Content-Length"):
                    val = resp.getheader(key)
                    if val:
                        self.send_header(key, val)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.URLError as exc:
            msg = f"Backend proxy error: {exc}".encode()
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(msg)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(msg)

    # ---- HTTP method handlers ----

    def do_GET(self):  # noqa: N802
        if self._should_proxy():
            self._proxy("GET")
        else:
            super().do_GET()

    def do_POST(self):  # noqa: N802
        if self._should_proxy():
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else None
            self._proxy("POST", body)
        else:
            self.send_response(405)
            self.end_headers()

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def log_message(self, fmt, *args):
        tag = "proxy" if self._should_proxy() else "static"
        sys.stderr.write(f"[{tag}] {self.address_string()} - {fmt % args}\n")


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8021
    bind = sys.argv[2] if len(sys.argv) > 2 else "0.0.0.0"

    web_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "web")
    os.chdir(web_dir)

    server = http.server.ThreadingHTTPServer((bind, port), ProxyHandler)
    print(f"Serving {web_dir} on http://{bind}:{port}")
    print(f"Proxying API requests to {BACKEND_ORIGIN}")
    server.serve_forever()


if __name__ == "__main__":
    main()
