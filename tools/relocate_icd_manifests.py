#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path


def rewrite_manifest(manifest_path: Path, install_libdir: Path) -> None:
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    icd = data.get("ICD")
    if not isinstance(icd, dict):
        return

    library_path = icd.get("library_path")
    if not isinstance(library_path, str) or not os.path.isabs(library_path):
        return

    library_file = Path(library_path).name
    target_library = install_libdir / library_file
    relative_library_path = os.path.relpath(target_library, manifest_path.parent)
    if library_path == relative_library_path:
        return

    icd["library_path"] = relative_library_path
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: relocate_icd_manifests.py <install_libdir> <install_icd_dir>", file=sys.stderr)
        return 2

    install_libdir = Path(argv[1])
    install_icd_dir = Path(argv[2])
    if not install_icd_dir.is_dir():
        return 0

    for manifest_path in sorted(install_icd_dir.glob("*.json")):
        rewrite_manifest(manifest_path, install_libdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
