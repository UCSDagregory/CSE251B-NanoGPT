#!/usr/bin/env python
import sys
from collections import deque

LOG_FILE = "run.log"
MAX_BYTES = 20 * 1024 * 1024   # keep last 20 MB

buf = deque()
size = 0

for chunk in iter(lambda: sys.stdin.buffer.readline(), b""):
    sys.stdout.buffer.write(chunk)
    sys.stdout.buffer.flush()

    buf.append(chunk)
    size += len(chunk)

    while size > MAX_BYTES and buf:
        old = buf.popleft()
        size -= len(old)

    with open(LOG_FILE, "wb") as f:
        f.writelines(buf)
