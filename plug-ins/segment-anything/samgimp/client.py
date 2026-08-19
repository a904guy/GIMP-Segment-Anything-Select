"""Client half of the plug-in to backend protocol.

The backend is a long-lived helper process holding the private Python
environment (torch, transformers and the model weights).  It outlives a
single plug-in invocation so the loaded model can be reused.

Messages are newline-delimited JSON over a loopback TCP socket, chosen for
identical behaviour on Linux, macOS and Windows.  The port and a random
per-daemon token are recorded in a state file, and the token is checked on
every request so other local processes cannot drive the daemon.
"""

import json
import os
import platform
import socket
import subprocess
import time

from . import env, log

PROTOCOL_VERSION = 1
CONNECT_TIMEOUT = 5.0


class BackendError(RuntimeError):
    """Raised when the backend reports a failure or cannot be reached."""


class BackendNotInstalled(BackendError):
    pass


# --------------------------------------------------------------------------
# Daemon state file
# --------------------------------------------------------------------------


def _read_state():
    try:
        with open(env.daemon_path(), "r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(state, dict):
        return None
    if "port" not in state or "token" not in state:
        return None
    return state


def _clear_state():
    try:
        os.remove(env.daemon_path())
    except OSError:
        pass


# --------------------------------------------------------------------------
# Connection
# --------------------------------------------------------------------------


class _Connection(object):
    """One request/response exchange over its own socket."""

    def __init__(self, port, timeout):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=CONNECT_TIMEOUT)
        self.sock.settimeout(timeout)
        self._buf = b""

    def send(self, payload):
        data = (json.dumps(payload) + "\n").encode("utf-8")
        self.sock.sendall(data)

    def read_message(self):
        while b"\n" not in self._buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                if not self._buf.strip():
                    return None
                chunk = b"\n"
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        line = line.strip()
        if not line:
            return None
        return json.loads(line.decode("utf-8"))

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class BackendClient(object):
    """Talks to the backend, starting it on demand."""

    def __init__(self):
        self._next_id = 1

    # -- process management ------------------------------------------------

    def _spawn(self):
        python = env.venv_python()
        if not os.path.isfile(python):
            raise BackendNotInstalled(
                "The Segment Anything runtime is not installed yet."
            )
        server = os.path.join(env.backend_dir(), "sam_server.py")
        if not os.path.isfile(server):
            raise BackendError("Backend script missing: %s" % server)

        env.ensure_dirs()
        _clear_state()

        settings = env.load_settings()
        # Keep every download and cache inside our own directory so that
        # uninstalling is a single folder removal.
        extra = {
            "HF_HOME": env.models_dir(),
            "HUGGINGFACE_HUB_CACHE": os.path.join(env.models_dir(), "hub"),
            "TORCH_HOME": os.path.join(env.models_dir(), "torch"),
            "SAM_GIMP_HOME": env.data_dir(),
        }
        if settings.get("hf_token"):
            extra["HF_TOKEN"] = settings["hf_token"]
        child_env = env.child_environment(extra)
        # Large, variably-sized allocations fragment the CUDA heap badly.
        child_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        cmd = [
            python, server,
            "--state-file", env.daemon_path(),
            "--idle-timeout", str(int(settings.get("idle_timeout_sec", 900))),
            "--log-dir", env.logs_dir(),
        ]

        kwargs = {
            "cwd": env.backend_dir(),
            "env": child_env,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if platform.system() == "Windows":
            # Detach so the helper survives GIMP closing this plug-in, and
            # never flash a console window at the user.
            kwargs["creationflags"] = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            )
        else:
            kwargs["start_new_session"] = True

        log.write("starting backend: %s" % " ".join(cmd))
        subprocess.Popen(cmd, **kwargs)

        deadline = time.time() + 60.0
        while time.time() < deadline:
            state = _read_state()
            if state:
                try:
                    conn = _Connection(state["port"], 5.0)
                    conn.close()
                    return state
                except OSError:
                    pass
            time.sleep(0.2)
        raise BackendError(
            "The backend did not start within 60 seconds. See %s"
            % env.display_path(os.path.join(env.logs_dir(), "backend.log"))
        )

    def _state(self, allow_spawn=True):
        state = _read_state()
        if state:
            try:
                conn = _Connection(state["port"], 5.0)
                conn.close()
                return state
            except OSError:
                _clear_state()
        if not allow_spawn:
            raise BackendError("Backend is not running.")
        return self._spawn()

    # -- requests ----------------------------------------------------------

    def call(self, op, params=None, timeout=600.0, on_progress=None, allow_spawn=True):
        """Run one operation.  ``on_progress(dict)`` is called for updates."""
        state = self._state(allow_spawn=allow_spawn)
        request = dict(params or {})
        request["op"] = op
        request["id"] = self._next_id
        request["token"] = state["token"]
        request["protocol"] = PROTOCOL_VERSION
        self._next_id += 1

        conn = _Connection(state["port"], timeout)
        try:
            conn.send(request)
            while True:
                message = conn.read_message()
                if message is None:
                    raise BackendError("Backend closed the connection unexpectedly.")
                if "progress" in message:
                    if on_progress:
                        on_progress(message["progress"])
                    continue
                if message.get("ok"):
                    return message.get("result", {})
                raise BackendError(message.get("error") or "Unknown backend error")
        except socket.timeout:
            raise BackendError("Timed out waiting for the backend (%.0fs)." % timeout)
        finally:
            conn.close()

    # -- convenience -------------------------------------------------------

    def ping(self, allow_spawn=True):
        return self.call("ping", timeout=30.0, allow_spawn=allow_spawn)

    def shutdown(self):
        try:
            self.call("shutdown", timeout=10.0, allow_spawn=False)
        except BackendError:
            pass
        _clear_state()


def is_running():
    state = _read_state()
    if not state:
        return False
    try:
        conn = _Connection(state["port"], 2.0)
        conn.close()
        return True
    except OSError:
        return False
