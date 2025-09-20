#!/usr/bin/env python3
"""
Lightweight task service that runs local jobs (atlas/qe) and exposes a simple dashboard.
No third-party deps; uses http.server. Intended for local development.

Endpoints:
- GET  /health             -> 200 ok
- GET  /tasks              -> JSON of tasks
- POST /run                -> submit job: {"software": "atlas|qe", "work_dir": "/abs/path"}
- GET  /dashboard          -> simple HTML auto-refresh page
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from pathlib import Path
import subprocess

CONFIG_PATH = Path("config/resources.yaml")


def load_local_binary(software: str) -> tuple[str, dict]:
    """Load local binary path and env for software from resources.yaml (new schema or simplified)."""
    import yaml
    data = yaml.safe_load(CONFIG_PATH.read_text())
    # New schema
    for res in data.get("resources", []) or []:
        if res.get("type", "local") == "local":
            sw = (res.get("software") or {}).get(software)
            if sw and sw.get("path"):
                env = sw.get("env") or {}
                return sw["path"], {str(k): str(v) for k, v in env.items()}
    # Simplified schema fallback
    sp = (data.get("software_paths") or {}).get(software)
    if sp:
        env = (data.get("software_configs") or {}).get(software, {}).get("execution", {}).get("environment_vars", {})
        return sp, {str(k): str(v) for k, v in env.items()}
    raise RuntimeError(f"No local binary configured for {software}")


def build_command(software: str, binary: str, work_dir: Path) -> str:
    if software == "qe":
        inp = "qe.in" if (work_dir / "qe.in").exists() else "job.in"
        return f"{binary} -in {inp} > job.out 2>&1"
    # atlas default
    inp = "atlas.in"
    return f"{binary} < {inp} > job.out 2>&1"


class TaskStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._tasks: dict[str, dict] = {}

    def add(self, t: dict):
        with self._lock:
            self._tasks[t["id"]] = t

    def update(self, task_id: str, **fields):
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].update(fields)

    def all(self) -> list[dict]:
        with self._lock:
            return list(self._tasks.values())


STORE = TaskStore()


class Runner(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def submit(self, task_id: str, pop: subprocess.Popen):
        with self._lock:
            self._procs[task_id] = pop

    def run(self):
        while True:
            time.sleep(1.0)
            with self._lock:
                for task_id, pop in list(self._procs.items()):
                    rc = pop.poll()
                    if rc is None:
                        continue
                    STORE.update(task_id, status="completed" if rc == 0 else "failed", returncode=int(rc), finished_at=time.time())
                    del self._procs[task_id]


RUNNER = Runner()
RUNNER.start()


class Handler(BaseHTTPRequestHandler):
    server_version = "TaskService/0.1"

    def _send(self, code: int, body: str, content_type: str = "text/plain"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body.encode("utf-8"))))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/health":
            self._send(200, "ok")
            return
        if p.path == "/tasks":
            body = json.dumps({"tasks": STORE.all()}, indent=2)
            self._send(200, body, "application/json")
            return
        if p.path == "/dashboard":
            self._send(200, self._dashboard_html(), "text/html")
            return
        self._send(404, "not found")

    def do_POST(self):
        p = urlparse(self.path)
        if p.path != "/run":
            self._send(404, "not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        software = payload.get("software", "atlas").lower()
        work_dir = Path(payload.get("work_dir", os.getcwd())).resolve()
        now = time.time()
        task_id = f"{software}_{work_dir.name}_{int(now)}"
        try:
            binary, extra_env = load_local_binary(software)
            cmd = build_command(software, binary, work_dir)
            env = os.environ.copy()
            env.update({str(k): str(v) for k, v in extra_env.items()})
            pop = subprocess.Popen(cmd, shell=True, cwd=str(work_dir), env=env)
            STORE.add({
                "id": task_id,
                "software": software,
                "work_dir": str(work_dir),
                "status": "running",
                "cmd": cmd,
                "started_at": now,
                "returncode": None,
            })
            RUNNER.submit(task_id, pop)
            self._send(200, json.dumps({"id": task_id}), "application/json")
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}), "application/json")

    @staticmethod
    def _dashboard_html() -> str:
        tasks = STORE.all()
        rows = []
        for t in tasks:
            rows.append(f"<tr><td>{t['id']}</td><td>{t['software']}</td><td>{t['status']}</td><td>{t.get('returncode')}</td><td><code>{t['work_dir']}</code></td></tr>")
        table = "\n".join(rows) if rows else "<tr><td colspan=5>No tasks</td></tr>"
        return f"""
<!doctype html>
<html><head><meta charset='utf-8'><title>Task Dashboard</title>
<meta http-equiv='refresh' content='2'>
<style>body{{font-family:sans-serif}} table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:6px}}</style>
</head><body>
<h3>Task Dashboard</h3>
<table>
<thead><tr><th>id</th><th>software</th><th>status</th><th>rc</th><th>work_dir</th></tr></thead>
<tbody>
{table}
</tbody>
</table>
</body></html>
"""


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Local task service (atlas/qe)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    httpd = HTTPServer((args.host, args.port), Handler)
    print(f"Task service listening on http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

