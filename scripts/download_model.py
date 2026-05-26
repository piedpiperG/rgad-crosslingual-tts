#!/usr/bin/env python3
"""Download RGAD cross-lingual TTS weights from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from rgad_crosslingual_tts.constants import DEFAULT_HF_REPO_ID, DEFAULT_MODEL_CACHE
from rgad_crosslingual_tts.download import download_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_CACHE)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()
    path = download_model(args.output_dir, repo_id=args.repo_id, revision=args.revision)
    print(path)


if __name__ == "__main__":
    main()
