"""Plug-in side of the Segment Anything integration for GIMP 3.

Only the standard library and GIMP's own GObject bindings may be imported
from here: this package runs inside GIMP's Python interpreter.  All of the
machine-learning work happens in the backend, which lives in its own
environment (see ``samgimp.setup_env``).
"""

__version__ = "1.0.0"
