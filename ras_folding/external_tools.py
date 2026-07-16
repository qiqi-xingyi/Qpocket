# Author: Yuqi Zhang
"""External-tool preflight (PULCHRA, OpenBabel, Vina).

Lightweight, explicit, no system-wide scanning. The lookup order is fixed:

    1. CLI / explicit ``explicit_path``
    2. environment variable (``env_var``)
    3. ``external_tools.json`` config (``config[tool_key]``)
    4. ``shutil.which(executable_name)``
    5. ``shutil.which(executable_name + ".exe")`` (Windows-friendly)

Anything beyond that is the user's responsibility — we DO NOT walk
``~/miniforge3``, conda envs, ``/usr/local`` trees, or any other
heuristic. If the tool is not in one of those four slots, we fail
fast with the exact list of paths we tried.

Public entry points:

* ``load_external_tools_config(path)`` — read a JSON config (may be None)
* ``resolve_external_tool(...)`` — resolve a single tool, never raises
* ``check_external_tool(path, version_args)`` — boot the binary briefly
* ``run_external_tools_preflight(...)`` — bundle for entry scripts
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


# ---------------------------------------------------------------------- #
# config                                                                 #
# ---------------------------------------------------------------------- #

def load_external_tools_config(
    config_path: Optional[Union[str, Path]],
) -> Dict[str, str]:
    """Read a JSON file mapping ``tool_key`` → path string.

    Returns ``{}`` when ``config_path`` is None or the file does not exist
    (a missing config is not an error — config is optional). Raises
    ``ValueError`` when the file exists but is malformed JSON.
    """
    if config_path is None:
        return {}
    p = Path(config_path)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"external_tools config at {p} is not valid JSON: {e}"
        ) from e
    if not isinstance(data, dict):
        raise ValueError(
            f"external_tools config at {p} must be a JSON object, "
            f"got {type(data).__name__}"
        )
    out: Dict[str, str] = {}
    for k, v in data.items():
        if v is None:
            continue
        out[str(k)] = str(v)
    return out


# ---------------------------------------------------------------------- #
# resolve                                                                #
# ---------------------------------------------------------------------- #

def _is_usable_file(path: Path) -> Optional[str]:
    """Return None when the file is good. Otherwise return a short reason."""
    if not path.exists():
        return "does not exist"
    if not path.is_file():
        return "exists but is not a regular file"
    if os.name == "nt":
        return None  # Windows: trust the .exe extension, skip x bit
    if not os.access(str(path), os.X_OK):
        return "exists but is NOT executable"
    return None


def resolve_external_tool(
    tool_key: str,
    executable_name: str,
    explicit_path: Optional[Union[str, Path]] = None,
    env_var: Optional[str] = None,
    config: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Resolve one external tool. Never raises.

    Returns a dict with:

    * ``tool``      — ``tool_key``
    * ``path``      — absolute path string when found, else ``None``
    * ``found``     — bool
    * ``source``    — one of ``"explicit" | "env" | "config" | "path" | "missing"``
    * ``attempted`` — list of ``{source, value, status}`` entries (every
      slot probed, including ones that were unset, so the user can see
      exactly what was tried)
    * ``error``     — short message when ``found is False``, else ``None``
    """
    attempts: List[Dict[str, Any]] = []

    def _try(source: str, value: Optional[str]) -> Optional[str]:
        """Probe one candidate. Returns the absolute path when usable,
        else None. Always records the attempt."""
        if not value:
            attempts.append(
                {"source": source, "value": None, "status": "<not set>"},
            )
            return None
        p = Path(value)
        why = _is_usable_file(p)
        if why is None:
            abs_path = str(p.resolve())
            attempts.append(
                {"source": source, "value": value, "status": "ok"},
            )
            return abs_path
        attempts.append(
            {"source": source, "value": value, "status": why},
        )
        return None

    # 1. explicit
    hit = _try("explicit", str(explicit_path) if explicit_path else None)
    if hit is not None:
        return _ok(tool_key, hit, "explicit", attempts)

    # 2. environment variable
    env_value = os.environ.get(env_var) if env_var else None
    hit = _try(f"env:{env_var}" if env_var else "env", env_value)
    if hit is not None:
        return _ok(tool_key, hit, "env", attempts)

    # 3. config[tool_key]
    config_value = (config or {}).get(tool_key)
    hit = _try(f"config[{tool_key}]", config_value)
    if hit is not None:
        return _ok(tool_key, hit, "config", attempts)

    # 4. PATH (executable_name)
    on_path = shutil.which(executable_name)
    hit = _try(f"PATH({executable_name})", on_path)
    if hit is not None:
        return _ok(tool_key, hit, "path", attempts)

    # 5. PATH (executable_name + ".exe")
    on_path_exe = shutil.which(executable_name + ".exe")
    hit = _try(f"PATH({executable_name}.exe)", on_path_exe)
    if hit is not None:
        return _ok(tool_key, hit, "path", attempts)

    return {
        "tool": tool_key,
        "path": None,
        "found": False,
        "source": "missing",
        "attempted": attempts,
        "error": (
            f"{tool_key!r} not found. Probed (in order): "
            + "; ".join(
                f"{a['source']}={a['value']!r} [{a['status']}]"
                for a in attempts
            )
        ),
    }


def _ok(
    tool_key: str,
    path: str,
    source: str,
    attempts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "tool": tool_key,
        "path": path,
        "found": True,
        "source": source,
        "attempted": attempts,
        "error": None,
    }


# ---------------------------------------------------------------------- #
# boot-check                                                             #
# ---------------------------------------------------------------------- #

def check_external_tool(
    path: Union[str, Path],
    version_args: Sequence[str] = (),
    timeout_sec: int = 10,
) -> Dict[str, Any]:
    """Boot the binary briefly to confirm it actually runs.

    We do NOT require returncode == 0 (PULCHRA prints help and exits 1
    when called with no args; obabel -V exits 0; vina --help exits 0;
    behaviour varies). We only require: it starts, it doesn't time out,
    and it produces SOMETHING on stdout or stderr.
    """
    cmd: List[str] = [str(path), *list(version_args)]
    out: Dict[str, Any] = {
        "path": str(path),
        "cmd": cmd,
        "ok": False,
        "returncode": None,
        "stdout_first_lines": [],
        "stderr_first_lines": [],
        "error": None,
    }
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=int(timeout_sec),
        )
    except FileNotFoundError as e:
        out["error"] = f"FileNotFoundError: {e}"
        return out
    except subprocess.TimeoutExpired as e:
        out["error"] = f"TimeoutExpired after {timeout_sec}s: {e}"
        return out
    except OSError as e:
        out["error"] = f"OSError: {e}"
        return out

    out["returncode"] = int(proc.returncode)
    out["stdout_first_lines"] = (proc.stdout or "").splitlines()[:5]
    out["stderr_first_lines"] = (proc.stderr or "").splitlines()[:5]
    produced_output = bool((proc.stdout or proc.stderr).strip())
    out["ok"] = produced_output
    if not produced_output:
        out["error"] = (
            f"binary started but produced no stdout/stderr "
            f"(returncode={proc.returncode}); cannot confirm health"
        )
    return out


# ---------------------------------------------------------------------- #
# preflight bundle                                                       #
# ---------------------------------------------------------------------- #

# Which boot-check argv to use for each tool. Empty tuple = run with no
# args. We chose argv that all three binaries handle without side effects:
# pulchra prints usage and exits non-zero; obabel -V prints version and
# exits 0; vina --help prints usage and exits 0.
_DEFAULT_VERSION_ARGS: Dict[str, Sequence[str]] = {
    "pulchra": (),
    "obabel": ("-V",),
    "vina": ("--help",),
}

_DEFAULT_EXECUTABLE_NAMES: Dict[str, str] = {
    "pulchra": "pulchra",
    "obabel": "obabel",
    "vina": "vina",
}

_DEFAULT_ENV_VARS: Dict[str, str] = {
    "pulchra": "PULCHRA_BIN",
    "obabel": "OBABEL_BIN",
    "vina": "VINA_BIN",
}


def run_external_tools_preflight(
    pulchra_bin: Optional[Union[str, Path]] = None,
    obabel_bin: Optional[Union[str, Path]] = None,
    vina_bin: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    require_pulchra: bool = True,
    require_docking: bool = True,
    boot_check: bool = True,
) -> Dict[str, Any]:
    """Resolve and (optionally) boot-check each tool. Never raises.

    ``require_pulchra``  — fails preflight if PULCHRA missing
    ``require_docking``  — fails preflight if obabel OR vina missing

    The returned dict is what entry scripts persist to
    ``external_tools_preflight.json``.
    """
    config = load_external_tools_config(config_path)

    resolved: Dict[str, Dict[str, Any]] = {}
    for key, explicit in (
        ("pulchra", pulchra_bin),
        ("obabel", obabel_bin),
        ("vina", vina_bin),
    ):
        r = resolve_external_tool(
            tool_key=key,
            executable_name=_DEFAULT_EXECUTABLE_NAMES[key],
            explicit_path=explicit,
            env_var=_DEFAULT_ENV_VARS[key],
            config=config,
        )
        if r["found"] and boot_check:
            r["boot_check"] = check_external_tool(
                r["path"], _DEFAULT_VERSION_ARGS[key],
            )
        resolved[key] = r

    missing: List[str] = []
    if require_pulchra and not resolved["pulchra"]["found"]:
        missing.append("pulchra")
    if require_docking:
        if not resolved["obabel"]["found"]:
            missing.append("obabel")
        if not resolved["vina"]["found"]:
            missing.append("vina")

    return {
        "config_path": str(config_path) if config_path else None,
        "config_loaded": dict(config),
        "require_pulchra": bool(require_pulchra),
        "require_docking": bool(require_docking),
        "pulchra": resolved["pulchra"],
        "obabel": resolved["obabel"],
        "vina": resolved["vina"],
        "missing": missing,
        "ok": not missing,
    }


__all__ = [
    "load_external_tools_config",
    "resolve_external_tool",
    "check_external_tool",
    "run_external_tools_preflight",
]
