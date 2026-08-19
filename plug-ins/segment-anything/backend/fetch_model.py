#!/usr/bin/env python3
"""Download model weights into the plug-in's private cache."""

import argparse
import os
import sys

# The original .pt checkpoints in these repos duplicate the safetensors
# weights that transformers actually loads, so they are skipped.
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model"]


def short(path):
    """Shorten the home directory to ``~`` so logs can be shared safely."""
    home = os.path.expanduser("~")
    if path.startswith(home + os.sep):
        return "~" + path[len(home):]
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="sam2")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    token = os.environ.get("HF_TOKEN") or None
    print("Downloading %s ..." % args.model)
    try:
        path = snapshot_download(
            repo_id=args.model,
            allow_patterns=ALLOW_PATTERNS,
            token=token,
        )
    except GatedRepoError:
        print(
            "\nERROR: %s is a gated repository.\n"
            "  1. Sign in at https://huggingface.co/%s and accept the licence.\n"
            "  2. Create a token at https://huggingface.co/settings/tokens\n"
            "  3. Paste the token into the plug-in's Setup dialog and try again."
            % (args.model, args.model),
            file=sys.stderr,
        )
        return 2
    except RepositoryNotFoundError:
        print("\nERROR: no such model repository: %s" % args.model, file=sys.stderr)
        return 2

    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    print("Weights ready at %s (%.1f MB)" % (short(path), total / 1048576.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
