#!/usr/bin/env python3
import os
import subprocess
import sys


def _find_web_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(os.path.dirname(here), "web")
    if os.path.isdir(candidate):
        return candidate
    pkg_dir = subprocess.check_output(["rospack", "find", "web_interface"]).decode().strip()
    return os.path.join(pkg_dir, "web")


WEB_DIR = _find_web_dir()
sys.path.insert(0, WEB_DIR)

import uvicorn
from app import app

PORT = 8090

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
