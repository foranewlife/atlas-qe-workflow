#!/usr/bin/env python3
"""
atlas_cli: convenience wrapper to run atlas in PWD via the local task service.

Usage:
  python scripts/atlas_cli.py atlas             # runs atlas in $PWD
  python scripts/atlas_cli.py serve [--port P]  # start service if not running
"""
import json
import os
import subprocess
import sys
import time
from urllib.request import urlopen, Request

SERVICE = ("127.0.0.1", 8765)


def svc_url(path: str) -> str:
    host, port = SERVICE
    return f"http://{host}:{port}{path}"


def ensure_service():
    try:
        with urlopen(svc_url("/health"), timeout=2) as resp:
            if resp.read().decode().strip() == "ok":
                return
    except Exception:
        pass
    # start service
    print("Starting task service...")
    subprocess.Popen([sys.executable, "scripts/task_service.py", "--host", SERVICE[0], "--port", str(SERVICE[1])])
    # wait for health
    for _ in range(20):
        try:
            with urlopen(svc_url("/health"), timeout=1) as resp:
                if resp.read().decode().strip() == "ok":
                    return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("service failed to start")


def run_atlas_here():
    ensure_service()
    payload = json.dumps({"software": "atlas", "work_dir": os.getcwd()}).encode()
    req = Request(svc_url("/run"), data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
        print(f"submitted: {data['id']}")
    print(f"dashboard: {svc_url('/dashboard')}")


def serve_only(port: int | None = None):
    if port:
        global SERVICE
        SERVICE = (SERVICE[0], port)
    ensure_service()
    print(f"service running at {svc_url('')}")


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "atlas":
        run_atlas_here()
    elif cmd == "serve":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else None
        serve_only(port)
    else:
        print(__doc__.strip())
        sys.exit(1)


if __name__ == "__main__":
    main()

