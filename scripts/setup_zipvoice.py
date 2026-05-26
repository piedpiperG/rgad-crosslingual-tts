#!/usr/bin/env python3
"""Clone the ZipVoice source tree pinned for this release."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from rgad_crosslingual_tts.constants import DEFAULT_ZIPVOICE_ROOT, ZIPVOICE_COMMIT


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ZIPVOICE_ROOT)
    parser.add_argument("--repo-url", default="https://github.com/k2-fsa/ZipVoice.git")
    parser.add_argument("--commit", default=ZIPVOICE_COMMIT)
    return parser


def main() -> None:
    args = get_parser().parse_args()
    if args.output_dir.exists():
        subprocess.run(["git", "fetch", "--all"], cwd=args.output_dir, check=True)
    else:
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", args.repo_url, str(args.output_dir)], check=True)
    subprocess.run(["git", "checkout", args.commit], cwd=args.output_dir, check=True)
    print(f"ZipVoice ready at {args.output_dir} ({args.commit})")


if __name__ == "__main__":
    main()
