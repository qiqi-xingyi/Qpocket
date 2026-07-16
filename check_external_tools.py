#!/usr/bin/env python
# Author: Yuqi Zhang
"""Standalone preflight for PULCHRA / OpenBabel / Vina.

Usage:

    python check_external_tools.py \\
        --config external_tools.json \\
        --json-out logs/external_tools_preflight.json

Exits 0 when every required tool is found and boot-checked. Exits 1
otherwise — prints exactly which tool is missing and which paths were
attempted, so the user knows what to put into ``external_tools.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from ras_folding.external_tools import run_external_tools_preflight


_HERE = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Resolve and boot-check external tools (PULCHRA, OpenBabel, "
            "Vina). Lookup order: explicit CLI > env var > config JSON > "
            "PATH. No system-wide scanning."
        ),
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to external_tools.json",
    )
    p.add_argument("--pulchra-bin", type=str, default=None)
    p.add_argument("--obabel-bin", type=str, default=None)
    p.add_argument("--vina-bin", type=str, default=None)
    p.add_argument(
        "--no-docking",
        action="store_true",
        help="Don't require obabel + vina (only PULCHRA).",
    )
    p.add_argument(
        "--no-pulchra",
        action="store_true",
        help="Don't require PULCHRA (e.g. if you only need docking).",
    )
    p.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Write the full preflight result to this JSON path.",
    )
    p.add_argument(
        "--no-boot-check",
        action="store_true",
        help=(
            "Skip the brief subprocess.run() health check. Useful in "
            "sandboxes where binaries cannot exec."
        ),
    )
    return p


def _print_result(result: dict) -> None:
    for key in ("pulchra", "obabel", "vina"):
        r = result[key]
        print(f"--- {key} ---")
        print(f"  found    = {r['found']}")
        print(f"  source   = {r['source']}")
        print(f"  path     = {r['path']}")
        if r.get("error"):
            print(f"  error    = {r['error']}")
        bc = r.get("boot_check")
        if bc is not None:
            print(f"  boot_ok  = {bc['ok']} (returncode={bc['returncode']})")
            if bc.get("error"):
                print(f"  boot_err = {bc['error']}")
            for line in (bc.get("stdout_first_lines") or [])[:2]:
                print(f"    stdout> {line}")
            for line in (bc.get("stderr_first_lines") or [])[:2]:
                print(f"    stderr> {line}")
        if not r["found"]:
            print("  attempted:")
            for a in r.get("attempted", []):
                print(
                    f"    - {a['source']}: {a['value']!r} [{a['status']}]"
                )

    print()
    print(
        f"missing = {result['missing']}  ok = {result['ok']}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config_path: Optional[Path] = None
    if args.config:
        cp = Path(args.config)
        config_path = cp if cp.is_absolute() else _HERE / cp

    result = run_external_tools_preflight(
        pulchra_bin=args.pulchra_bin,
        obabel_bin=args.obabel_bin,
        vina_bin=args.vina_bin,
        config_path=config_path,
        require_pulchra=not bool(args.no_pulchra),
        require_docking=not bool(args.no_docking),
        boot_check=not bool(args.no_boot_check),
    )

    _print_result(result)

    if args.json_out:
        out_path = Path(args.json_out)
        if not out_path.is_absolute():
            out_path = _HERE / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result, indent=2), encoding="utf-8",
        )
        print(f"\nwrote preflight JSON: {out_path}")

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
