"""pysimcraft — drive SimCraft from Python.

Every SimCraft tool is available as a method, with the tool's description as its docstring:

    import pysimcraft

    with pysimcraft.connect() as sim:            # the open SimCraft window, or a headless server
        sim.new_model()
        sim.add_box(name="Beam", size=[100, 10, 8])
        sim.add_fixed_support(faces=["Beam.xmin"])
        sim.add_force(faces=["Beam.xmax"], vector=[0, 0, -100])
        r = sim.solve()
        print(r.data["fields"]["von_mises"]["max"], "MPa")
        sim.render_image(field="von_mises").save_image("beam.png")

`connect()` talks to a running SimCraft (GUI or `simcraft serve`); if none is running it starts a
headless server and stops it again when the `with` block ends. Only the Python standard library is
used.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

__all__ = ["SimCraft", "Result", "SimCraftError", "connect", "find_binary"]
__version__ = "0.1.0"

DEFAULT_PORT = 47800
MAX_RESPONSE_BYTES = 64 * 1024 * 1024


class SimCraftError(RuntimeError):
    """A SimCraft tool reported an error (bad arguments, unsolvable model, …)."""


class Result:
    """Outcome of one tool call.

    - ``text``: SimCraft's reply as text
    - ``data``: the reply parsed as JSON (dict/list) when it is JSON, else ``None``
    - ``image``: PNG bytes for image tools (``render_image``, ``screenshot_gui``), else ``None``
    """

    def __init__(self, text: str, image: Optional[bytes]):
        self.text = text
        self.image = image
        try:
            self.data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            self.data = None

    def __getitem__(self, key):
        if self.data is None:
            raise KeyError(f"result is plain text, not JSON: {self.text[:80]!r}")
        return self.data[key]

    def save_image(self, path: str | os.PathLike) -> Path:
        if self.image is None:
            raise SimCraftError("this result has no image")
        p = Path(path)
        p.write_bytes(self.image)
        return p

    def __repr__(self) -> str:
        kind = "json" if self.data is not None else "text"
        img = f", image {len(self.image)} B" if self.image else ""
        return f"<Result {kind}{img}: {self.text[:60]!r}>"


def _token_path() -> Path:
    if os.environ.get("SIMCRAFT_TOKEN_FILE"):
        return Path(os.environ["SIMCRAFT_TOKEN_FILE"])
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return Path(base) / "simcraft" / "token"


def find_binary() -> Optional[str]:
    """Locate the simcraft executable: $SIMCRAFT_BIN, PATH, then ~/simcraft/target/release."""
    for cand in (os.environ.get("SIMCRAFT_BIN"), shutil.which("simcraft"), os.path.expanduser("~/simcraft/target/release/simcraft")):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


class SimCraft:
    """Connection to a running SimCraft (window or headless server)."""

    def __init__(self, host: str = "127.0.0.1", port: Optional[int] = None, timeout: float = 3600.0, _server: Optional[subprocess.Popen] = None):
        self.host = host
        self.port = port or int(os.environ.get("SIMCRAFT_PORT", DEFAULT_PORT))
        self._server = _server
        self._sock = socket.create_connection((host, self.port), timeout=timeout)
        self._file = self._sock.makefile("rwb")
        try:
            token = _token_path().read_text().strip()
        except OSError as e:
            raise SimCraftError(f"cannot read the SimCraft token at {_token_path()} ({e}); start SimCraft once to create it") from e
        reply = self._roundtrip({"auth": token})
        if not reply.get("ok"):
            raise SimCraftError(reply.get("text", "SimCraft refused the connection"))
        self._tools = {t["name"]: t for t in self.call("list_tools").data}

    # -- protocol -------------------------------------------------------------------------
    def _roundtrip(self, obj: dict) -> dict:
        self._file.write((json.dumps(obj) + "\n").encode())
        self._file.flush()
        line = self._file.readline(MAX_RESPONSE_BYTES + 1)
        if not line:
            raise SimCraftError("SimCraft closed the connection")
        if len(line) > MAX_RESPONSE_BYTES or not line.endswith(b"\n"):
            raise SimCraftError("SimCraft sent an oversized or truncated response")
        return json.loads(line)

    def call(self, tool: str, **args: Any) -> Result:
        """Call any tool by name; keyword arguments become the tool's JSON arguments."""
        reply = self._roundtrip({"tool": tool, "args": args})
        if not reply.get("ok"):
            raise SimCraftError(f"{tool}: {reply.get('text', 'failed')}")
        img = reply.get("image_png_b64")
        return Result(reply.get("text", ""), base64.b64decode(img) if img else None)

    # -- tools as methods ---------------------------------------------------------------
    def tools(self) -> list[str]:
        """Names of all available tools."""
        return sorted(self._tools)

    def __getattr__(self, name: str):
        tools = self.__dict__.get("_tools") or {}
        if name not in tools:
            raise AttributeError(f"SimCraft has no tool {name!r}; see .tools()")
        spec = tools[name]

        def method(**args: Any) -> Result:
            return self.call(name, **args)

        props = spec.get("inputSchema", {}).get("properties", {})
        required = set(spec.get("inputSchema", {}).get("required", []))
        lines = [spec.get("description", ""), ""]
        for k, v in props.items():
            desc = v.get("description", "")
            typ = v.get("type", "")
            lines.append(f"    {k}{' (required)' if k in required else ''}: {typ} {desc}".rstrip())
        method.__name__ = name
        method.__doc__ = "\n".join(lines)
        return method

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self.__dict__.get("_tools", {})))

    # -- lifecycle ------------------------------------------------------------------------
    def close(self) -> None:
        try:
            self._file.close()
            self._sock.close()
        finally:
            if self._server is not None:
                self._server.terminate()
                try:
                    self._server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._server.kill()
                self._server = None

    def __enter__(self) -> "SimCraft":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


def connect(host: str = "127.0.0.1", port: Optional[int] = None, start_server: bool = True, binary: Optional[str] = None, timeout: float = 3600.0) -> SimCraft:
    """Connect to SimCraft, starting a headless server (`simcraft serve`) if none is running."""
    port = port or int(os.environ.get("SIMCRAFT_PORT", DEFAULT_PORT))
    if _listening(host, port):
        return SimCraft(host, port, timeout)
    if not start_server:
        raise SimCraftError(f"no SimCraft listening on {host}:{port}")
    exe = binary or find_binary()
    if not exe:
        raise SimCraftError("simcraft executable not found; set SIMCRAFT_BIN or put simcraft on PATH")
    proc = subprocess.Popen([exe, "serve", "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        if _listening(host, port):
            return SimCraft(host, port, timeout, _server=proc)
        if proc.poll() is not None:
            raise SimCraftError(f"simcraft serve exited with code {proc.returncode}")
        time.sleep(0.1)
    proc.terminate()
    raise SimCraftError("timed out waiting for the SimCraft server to start")
