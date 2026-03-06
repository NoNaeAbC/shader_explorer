#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def main():
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <stamp_out> <cmd> [args...]", file=sys.stderr)
        return 2
    stamp = Path(sys.argv[1])
    cmd = sys.argv[2:]
    subprocess.check_call(cmd)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text("ok\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
