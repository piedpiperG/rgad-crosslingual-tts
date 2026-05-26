#!/usr/bin/env python3
"""Single-utterance inference CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from rgad_crosslingual_tts.constants import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_STEP,
    DEFAULT_PROMPT_SECONDS,
    DEFAULT_SPEED,
    DEFAULT_T_SHIFT,
)
from rgad_crosslingual_tts.infer import synthesize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-wav", required=True, type=Path)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-wav", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--zipvoice-root", type=Path, default=None)
    parser.add_argument("--prompt-seconds", type=float, default=DEFAULT_PROMPT_SECONDS)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--num-step", type=int, default=DEFAULT_NUM_STEP)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--t-shift", type=float, default=DEFAULT_T_SHIFT)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num-thread", type=int, default=1)
    parser.add_argument("--vocoder-path", type=Path, default=None)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    output = synthesize(
        prompt_wav=args.prompt_wav,
        text=args.text,
        output_wav=args.output_wav,
        model_dir=args.model_dir,
        zipvoice_root=args.zipvoice_root,
        prompt_seconds=args.prompt_seconds,
        prompt_text=args.prompt_text,
        speed=args.speed,
        num_step=args.num_step,
        guidance_scale=args.guidance_scale,
        t_shift=args.t_shift,
        gpu=args.gpu,
        num_thread=args.num_thread,
        vocoder_path=args.vocoder_path,
        download=not args.no_download,
    )
    print(output)


if __name__ == "__main__":
    main()
