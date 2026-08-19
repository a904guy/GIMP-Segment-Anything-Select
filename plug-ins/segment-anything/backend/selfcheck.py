#!/usr/bin/env python3
"""Verify the private environment can actually run the models.

Run by the installer as the last step before downloading weights, so that
a broken PyTorch install is reported while the user is still looking at
the setup dialog rather than halfway through a 3 GB download.
"""

import sys


def main():
    print("python %s" % sys.version.split()[0])
    print("executable %s" % sys.executable)

    import numpy
    print("numpy %s" % numpy.__version__)

    import torch
    print("torch %s" % torch.__version__)
    if torch.cuda.is_available():
        print("cuda available: %s (%d device(s))"
              % (torch.version.cuda, torch.cuda.device_count()))
        print("gpu 0: %s" % torch.cuda.get_device_name(0))
    else:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            print("apple metal (mps) available")
        else:
            print("no gpu detected - running on cpu")

    import transformers
    print("transformers %s" % transformers.__version__)

    missing = []
    for name in ("Sam2Model", "Sam2Processor"):
        if not hasattr(transformers, name):
            missing.append(name)
    if missing:
        print("WARNING: transformers is missing %s; SAM 2 will not work"
              % ", ".join(missing))
    else:
        print("SAM 2 support present")
    if hasattr(transformers, "Sam3Model"):
        print("SAM 3 support present")
    else:
        print("WARNING: this transformers build has no SAM 3 support")

    from PIL import Image
    print("pillow ok (%s)" % Image.__name__)

    # A real tensor round-trip catches broken CUDA/driver pairings that a
    # plain import does not.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.ones((64, 64), device=device)
    assert float((tensor @ tensor).sum()) == 64 * 64 * 64
    print("tensor maths ok on %s" % device)
    print("SELFCHECK OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
