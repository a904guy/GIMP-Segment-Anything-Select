"""Tiny logging helper shared by the plug-in side."""

import datetime
import os
import sys
import traceback

from . import env

_MAX_BYTES = 512 * 1024


def log_path():
    return os.path.join(env.logs_dir(), "plugin.log")


def write(message):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = "[%s] %s" % (stamp, message)
    try:
        env.ensure_dirs()
        path = log_path()
        if os.path.exists(path) and os.path.getsize(path) > _MAX_BYTES:
            try:
                os.replace(path, path + ".1")
            except OSError:
                pass
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass
    print(line, file=sys.stderr)


def exception(message):
    write(message + "\n" + traceback.format_exc())
