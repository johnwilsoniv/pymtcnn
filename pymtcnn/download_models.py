"""
pymtcnn-download-models: download the OpenFace model files pymtcnn needs.

Usage:
    pymtcnn-download-models                   # shows the license summary, asks you to type YES
    pymtcnn-download-models --accept-license  # accepts the license without asking
    python -m pymtcnn.download_models         # same, if the command is not found

The files come from OpenFace's official GitHub repository, are checked against
SHA-256 checksums, converted into the ONNX and Core ML models pymtcnn uses, and
stored in a cache shared with pyclnf and pyfaceau (see ``pymtcnn.models``).
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import models

LICENSE_SUMMARY = f"""\
pymtcnn uses face-detection model files from OpenFace 2.2.0. They are not part
of pymtcnn: they belong to Carnegie Mellon University and are downloaded from
OpenFace's official GitHub page, then checked and converted on this computer.

OpenFace license, in short (the full text is what counts):
  - Use only for academic or non-profit, non-commercial research.
  - Do not share, sell or give others access to these files.
  - Commercial use needs a separate license from Carnegie Mellon University.

Full license: {models.LICENSE_URL}
"""


def _say(text: str = "", end: str = "\n") -> None:
    try:
        print(text, end=end, flush=True)
    except (BrokenPipeError, OSError):
        pass


class _ProgressBar:
    """Single-line progress for a terminal, one line per file otherwise."""

    def __init__(self, label: str):
        self.label = label
        self.tty = sys.stdout.isatty()
        self.last_name = None
        self.last_pct = -1

    def __call__(self, *args, **kwargs) -> None:
        # Tolerant of other packages' callbacks: (done, total, name) is expected.
        try:
            done, total = int(args[0]), int(args[1])
            name = str(args[2]) if len(args) > 2 else ""
        except (IndexError, TypeError, ValueError):
            return
        if total <= 0:
            return
        pct = min(100, int(done * 100 / total))
        if self.tty:
            if pct != self.last_pct:
                bar = "#" * (pct // 4)
                _say(f"\r  {self.label}: [{bar:<25}] {pct:3d}%  "
                     f"{models._human_size(done)} of {models._human_size(total)}", end="")
                if done >= total:
                    _say()
        else:
            if name and name != self.last_name:
                _say(f"  downloading {name.rsplit('/', 1)[-1]}")
            elif done >= total and self.last_pct < 100:
                _say(f"  {self.label}: done ({models._human_size(total)})")
        self.last_name = name or self.last_name
        self.last_pct = pct


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog=models.DOWNLOAD_COMMAND,
        description="Download OpenFace 2.2.0's MTCNN face-detection files and prepare "
                    "the models pymtcnn uses.")
    parser.add_argument("--accept-license", action="store_true",
                        help="accept the OpenFace license without being asked")
    parser.add_argument("--cache-dir", metavar="FOLDER",
                        help=f"use this folder instead of the default (same as {models.ENV_MODELS_DIR})")
    args = parser.parse_args(argv)

    root = models.models_root(args.cache_dir)
    _say("OpenFace model files for pymtcnn")
    _say("================================")
    _say(LICENSE_SUMMARY)

    accepted = models.license_accepted(args.accept_license)
    if accepted:
        _say("License accepted (--accept-license or OPENFACE_MODELS_ACCEPT_LICENSE=1).\n")
    else:
        try:
            answer = input("Type YES to accept the OpenFace license and download the files: ")
        except EOFError:
            answer = ""
        except KeyboardInterrupt:
            _say("\nCancelled. Nothing was downloaded.")
            return 130
        if answer.strip().upper() != "YES":
            _say("\nNothing was downloaded, because the license was not accepted.")
            _say(f"To accept it without being asked, run:  {models.DOWNLOAD_COMMAND} --accept-license")
            return 1
        _say()

    _say(f"Saving to: {root}")
    try:
        path = models.ensure_models(accept_license=True, cache_dir=args.cache_dir,
                                    progress=_ProgressBar("pymtcnn"))
    except KeyboardInterrupt:
        _say("\nCancelled. Run the command again to finish; files already checked are kept.")
        return 130
    except models.ModelDownloadError as e:
        _say(f"\nThe download did not finish.\n{e}")
        return 1
    except Exception as e:
        _say(f"\nSomething went wrong: {e}")
        return 1

    _say("\nDone. The OpenFace model files are installed, checked and converted.")
    _say(f"  pymtcnn: ready in {path}")
    _say("\nYou can now use pymtcnn.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
