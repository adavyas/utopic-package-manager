import json
import os
import shutil
import subprocess
import ssl
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


class BrokenDownloadServer(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"partial gguf"
        self.send_response(200)
        self.send_header("content-type", "application/octet-stream")
        self.send_header("content-length", str(len(body) + 1024))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self.connection.close()

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


def test_bundled_chat_removes_partial_model_after_download_failure(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("openssl is not installed")

    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-subj",
            "/CN=127.0.0.1",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
            "-days",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), BrokenDownloadServer)
    except PermissionError as exc:
        pytest.skip(f"localhost bind is unavailable in this environment: {exc}")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        models_dir = tmp_path / "models"
        catalog = tmp_path / "models.json"
        catalog.write_text(
            json.dumps(
                [
                    {
                        "id": "broken",
                        "name": "Broken",
                        "family": "test",
                        "filename": "broken.gguf",
                        "url": f"https://127.0.0.1:{server.server_port}/broken.gguf",
                        "size": "1 KiB",
                        "recommended": True,
                        "description": "Broken test download",
                    }
                ]
            ),
            encoding="utf-8",
        )

        completed = subprocess.run(
            [node, str(CHAT_SCRIPT), "broken"],
            capture_output=True,
            text=True,
            timeout=15,
            env={
                **os.environ,
                "NODE_TLS_REJECT_UNAUTHORIZED": "0",
                "UTOPIC_MODELS_CATALOG": str(catalog),
                "UTOPIC_MODELS_DIR": str(models_dir),
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert completed.returncode == 1
    assert not (models_dir / "broken.gguf.partial").exists()
    assert not (models_dir / "broken.gguf").exists()


def test_bundled_chat_waits_for_started_server_to_exit(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")

    model = tmp_path / "model.gguf"
    model.write_text("fake model", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    state_file = tmp_path / "server-state.jsonl"
    fake_server = bin_dir / "utopic_server"
    fake_server.write_text(
        f"""#!{node}
const fs = require("node:fs");
const http = require("node:http");

function argValue(name, fallback) {{
  const index = process.argv.indexOf(name);
  return index >= 0 && index + 1 < process.argv.length ? process.argv[index + 1] : fallback;
}}

const host = argValue("--host", "127.0.0.1");
const port = Number(argValue("--port", "8910"));
const stateFile = {json.dumps(str(state_file))};
function write(event) {{
  fs.appendFileSync(stateFile, JSON.stringify({{ event, pid: process.pid }}) + "\\n");
}}

const server = http.createServer((req, res) => {{
  if (req.method === "GET" && req.url === "/health") {{
    res.writeHead(200, {{ "content-type": "application/json" }});
    res.end(JSON.stringify({{ status: "ok" }}));
    return;
  }}
  if (req.method === "POST" && req.url === "/v1/chat/completions") {{
    req.resume();
    res.writeHead(200, {{ "content-type": "application/json" }});
    res.end(JSON.stringify({{ choices: [{{ message: {{ role: "assistant", content: "fake answer" }} }}] }}));
    return;
  }}
  res.writeHead(404);
  res.end();
}});

server.listen(port, host, () => write("listening"));
process.on("SIGTERM", () => {{
  write("term");
  setTimeout(() => {{
    write("exit");
    server.close(() => process.exit(0));
  }}, 400);
}});
""",
        encoding="utf-8",
    )
    fake_server.chmod(0o755)
    try:
        port_server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    except PermissionError as exc:
        pytest.skip(f"localhost bind is unavailable in this environment: {exc}")
    else:
        port = port_server.server_port
        port_server.server_close()

    completed = subprocess.run(
        [
            node,
            str(CHAT_SCRIPT),
            "--model",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        input="/exit\n",
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
        env={**os.environ, "UTOPIC_BIN_DIR": str(bin_dir)},
    )

    assert "OpenAI-compatible URL:" in completed.stdout
    events = [json.loads(line)["event"] for line in state_file.read_text(encoding="utf-8").splitlines()]
    assert events == ["listening", "term", "exit"]


def test_bundled_chat_preserves_equals_in_model_flag_value(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")

    model = tmp_path / "model=with-equals.gguf"
    model.write_text("fake model", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    state_file = tmp_path / "server-state.jsonl"
    fake_server = bin_dir / "utopic_server"
    fake_server.write_text(
        f"""#!{node}
const fs = require("node:fs");
const http = require("node:http");

function argValue(name, fallback) {{
  const index = process.argv.indexOf(name);
  return index >= 0 && index + 1 < process.argv.length ? process.argv[index + 1] : fallback;
}}

const host = argValue("--host", "127.0.0.1");
const port = Number(argValue("--port", "8910"));
const stateFile = {json.dumps(str(state_file))};
fs.appendFileSync(stateFile, JSON.stringify({{ event: "argv", argv: process.argv }}) + "\\n");

const server = http.createServer((req, res) => {{
  if (req.method === "GET" && req.url === "/health") {{
    res.writeHead(200, {{ "content-type": "application/json" }});
    res.end(JSON.stringify({{ status: "ok" }}));
    return;
  }}
  res.writeHead(404);
  res.end();
}});

server.listen(port, host);
process.on("SIGTERM", () => server.close(() => process.exit(0)));
""",
        encoding="utf-8",
    )
    fake_server.chmod(0o755)
    try:
        port_server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    except PermissionError as exc:
        pytest.skip(f"localhost bind is unavailable in this environment: {exc}")
    else:
        port = port_server.server_port
        port_server.server_close()

    subprocess.run(
        [
            node,
            str(CHAT_SCRIPT),
            f"--model={model}",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        input="/exit\n",
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
        env={**os.environ, "UTOPIC_BIN_DIR": str(bin_dir)},
    )

    first_event = json.loads(state_file.read_text(encoding="utf-8").splitlines()[0])
    assert first_event["argv"][first_event["argv"].index("-m") + 1] == str(model)
