#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn


CORE_SYSTEM_LIBS = frozenset({
    "linux-vdso.so.1",
    "ld-linux-x86-64.so.2",
    "ld-linux-aarch64.so.1",
    "libBrokenLocale.so.1",
    "libanl.so.1",
    "libc.so.6",
    "libdl.so.2",
    "libm.so.6",
    "libpthread.so.0",
    "libresolv.so.2",
    "librt.so.1",
    "libutil.so.1",
})

LDD_WITH_PATH_RE = re.compile(r"^\s*(?P<name>\S+)\s+=>\s+(?P<path>/\S+)\s+\(")
LDD_DIRECT_PATH_RE = re.compile(r"^\s*(?P<path>/\S+)\s+\(")
LDD_MISSING_RE = re.compile(r"^\s*(?P<name>\S+)\s+=>\s+not found$")


@dataclass(frozen=True)
class ExternalDependency:
    owner: Path
    name: str
    path: Path | None


def fail(message: str) -> NoReturn:
    raise SystemExit(message)


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout + result.stderr


def is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
        return True
    except (FileNotFoundError, ValueError):
        return False


def candidate_libdirs(package_root: Path) -> list[Path]:
    candidates: list[Path] = []

    icd_dir = package_root / "share" / "vulkan" / "icd.d"
    for manifest_path in sorted(icd_dir.glob("*.json")):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        library_path = data.get("ICD", {}).get("library_path")
        if not isinstance(library_path, str):
            continue
        candidate = (manifest_path.parent / library_path).resolve()
        if candidate.exists():
            candidates.append(candidate.parent)

    for pattern in ("lib", "lib64", "lib/*-linux-gnu", "usr/lib", "usr/lib64"):
        for match in sorted(package_root.glob(pattern)):
            if match.is_dir():
                candidates.append(match.resolve())

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def detect_bundle_libdir(package_root: Path) -> Path:
    for candidate in candidate_libdirs(package_root):
        if any(candidate.glob("libvulkan*.so*")) or any(candidate.glob("libvulkan_*.so*")):
            return candidate
    for candidate in candidate_libdirs(package_root):
        if any(candidate.iterdir()):
            return candidate
    fail(f"failed to detect packaged library directory under {package_root}")


def iter_package_elfs(package_root: Path) -> list[Path]:
    elfs: list[Path] = []
    seen_resolved: set[Path] = set()
    for candidate in sorted(package_root.rglob("*")):
        if not candidate.exists():
            continue
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        if not resolved.is_file() or not path_is_within(resolved, package_root):
            continue
        if resolved in seen_resolved:
            continue
        if is_elf(resolved):
            elfs.append(resolved)
            seen_resolved.add(resolved)
    return elfs


def ldd_env(bundle_libdir: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("LD_LIBRARY_PATH")
    if existing:
        env["LD_LIBRARY_PATH"] = f"{bundle_libdir}:{existing}"
    else:
        env["LD_LIBRARY_PATH"] = str(bundle_libdir)
    return env


def list_external_deps(package_root: Path, bundle_libdir: Path) -> tuple[list[ExternalDependency], list[ExternalDependency]]:
    missing: list[ExternalDependency] = []
    external: list[ExternalDependency] = []
    env = ldd_env(bundle_libdir)

    for elf_path in iter_package_elfs(package_root):
        output = run(["ldd", str(elf_path)], env=env)
        for line in output.splitlines():
            line = line.strip()
            if not line or line.startswith("statically linked"):
                continue

            missing_match = LDD_MISSING_RE.match(line)
            if missing_match:
                missing.append(ExternalDependency(elf_path, missing_match.group("name"), None))
                continue

            with_path_match = LDD_WITH_PATH_RE.match(line)
            if with_path_match:
                name = with_path_match.group("name")
                dep_path = Path(with_path_match.group("path")).resolve()
            else:
                direct_path_match = LDD_DIRECT_PATH_RE.match(line)
                if not direct_path_match:
                    continue
                dep_path = Path(direct_path_match.group("path")).resolve()
                name = dep_path.name

            if name in CORE_SYSTEM_LIBS or dep_path.name in CORE_SYSTEM_LIBS:
                continue
            if path_is_within(dep_path, package_root):
                continue

            external.append(ExternalDependency(elf_path, name, dep_path))

    return missing, external


def ensure_symlink(path: Path, target: str) -> None:
    if path.is_symlink():
        if os.readlink(path) == target:
            return
        fail(f"refusing to replace symlink {path} -> {os.readlink(path)} with {target}")
    if path.exists():
        return
    path.symlink_to(target)


def read_soname(path: Path) -> str | None:
    result = subprocess.run(
        ["patchelf", "--print-soname", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    soname = result.stdout.strip()
    return soname or None


def copy_dependency(dep_path: Path, bundle_libdir: Path) -> bool:
    dep_path = dep_path.resolve(strict=True)
    source_path = Path(dep_path)
    resolved_path = source_path.resolve(strict=True)

    bundle_libdir.mkdir(parents=True, exist_ok=True)

    copied_new = False
    dest_real = bundle_libdir / resolved_path.name
    if not dest_real.exists():
        shutil.copy2(resolved_path, dest_real)
        copied_new = True

    if source_path.name != resolved_path.name:
        ensure_symlink(bundle_libdir / source_path.name, resolved_path.name)

    soname = read_soname(dest_real)
    if soname and soname not in {resolved_path.name, source_path.name}:
        ensure_symlink(bundle_libdir / soname, resolved_path.name)

    return copied_new


def compute_rpath(elf_path: Path, bundle_libdir: Path) -> str:
    relative = os.path.relpath(bundle_libdir, elf_path.parent)
    if relative == ".":
        return "$ORIGIN"
    return "$ORIGIN/" + relative.replace(os.sep, "/")


def patch_rpaths(package_root: Path, bundle_libdir: Path) -> None:
    for elf_path in iter_package_elfs(package_root):
        rpath = compute_rpath(elf_path, bundle_libdir)
        subprocess.run(["patchelf", "--set-rpath", rpath, str(elf_path)], check=True)


def bundle_deps(package_root: Path, bundle_libdir: Path) -> int:
    bundle_count = 0
    while True:
        missing, external = list_external_deps(package_root, bundle_libdir)
        if missing:
            details = "\n".join(
                f"{dep.owner}: missing {dep.name}"
                for dep in missing
            )
            fail(f"missing shared library dependencies:\n{details}")

        copied_any = False
        for dep in external:
            assert dep.path is not None
            if copy_dependency(dep.path, bundle_libdir):
                bundle_count += 1
                copied_any = True

        patch_rpaths(package_root, bundle_libdir)
        if not copied_any:
            return bundle_count


def validate_no_external_deps(package_root: Path, bundle_libdir: Path) -> None:
    missing, external = list_external_deps(package_root, bundle_libdir)
    if missing or external:
        messages: list[str] = []
        for dep in missing:
            messages.append(f"{dep.owner}: missing {dep.name}")
        for dep in external:
            assert dep.path is not None
            messages.append(f"{dep.owner}: external {dep.name} => {dep.path}")
        fail("packaged ELF closure check failed:\n" + "\n".join(messages))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bundle non-core shared library dependencies into a packaged install tree.",
    )
    parser.add_argument("package_root", type=Path)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate dependency closure without copying any libraries.",
    )
    parser.add_argument(
        "--libdir",
        type=Path,
        help="Override the bundled library directory relative to the package root.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv[1:])
    package_root = args.package_root.resolve()
    if not package_root.is_dir():
        fail(f"package root does not exist: {package_root}")

    bundle_libdir = (package_root / args.libdir).resolve() if args.libdir else detect_bundle_libdir(package_root)
    if not path_is_within(bundle_libdir, package_root):
        fail(f"bundle libdir must stay within the package root: {bundle_libdir}")

    if args.check_only:
        validate_no_external_deps(package_root, bundle_libdir)
        print(f"validated packaged runtime closure under {package_root}")
        return 0

    patch_rpaths(package_root, bundle_libdir)
    bundled = bundle_deps(package_root, bundle_libdir)
    validate_no_external_deps(package_root, bundle_libdir)
    print(f"bundled {bundled} external shared libraries into {bundle_libdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
