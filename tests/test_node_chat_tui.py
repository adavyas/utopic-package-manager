import json
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


CHAT_SCRIPT = Path(__file__).resolve().parents[1] / "python" / "utopic" / "node" / "utopic-chat.js"


class FakeOpenAIServer(BaseHTTPRequestHandler):
    requests = []
    paths = []

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length)
        self.__class__.paths.append(self.path)
        self.__class__.requests.append(json.loads(raw.decode("utf-8")))
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "hello from fake utopic",
                        }
                    }
                ]
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


@pytest.fixture()
def fake_openai_server():
    FakeOpenAIServer.requests = []
    FakeOpenAIServer.paths = []
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAIServer)
    except PermissionError as exc:
        pytest.skip(f"localhost bind is unavailable in this environment: {exc}")

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", FakeOpenAIServer.requests, FakeOpenAIServer.paths
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_bundled_chat_help_runs_without_server():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")

    completed = subprocess.run(
        [node, str(CHAT_SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "usage: utopic chat" in completed.stdout
    assert "utopic chat dream-7b-q4" in completed.stdout


def test_bundled_chat_posts_messages_to_openai_compatible_server(fake_openai_server):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    base_url, requests, paths = fake_openai_server

    completed = subprocess.run(
        [node, str(CHAT_SCRIPT), "--server", base_url],
        input="hi\n/exit\n",
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert "OpenAI-compatible URL:" in completed.stdout
    assert "hello from fake utopic" in completed.stdout
    assert paths == ["/v1/chat/completions"]
    assert requests == [
        {
            "model": "utopic",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 512,
            "temperature": 0,
        }
    ]


def test_bundled_chat_accepts_openai_compatible_server_url(fake_openai_server):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    base_url, requests, paths = fake_openai_server

    completed = subprocess.run(
        [node, str(CHAT_SCRIPT), "--server", f"{base_url}/v1/chat/completions"],
        input="hi\n/exit\n",
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert f"OpenAI-compatible URL: {base_url}/v1/chat/completions" in completed.stdout
    assert "hello from fake utopic" in completed.stdout
    assert paths == ["/v1/chat/completions"]
    assert requests == [
        {
            "model": "utopic",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 512,
            "temperature": 0,
        }
    ]
