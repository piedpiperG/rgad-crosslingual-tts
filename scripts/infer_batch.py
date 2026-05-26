#!/usr/bin/env python3
"""Batch inference CLI.

Input TSV columns:
  wav_name<TAB>prompt_wav<TAB>target_text

Optional fourth column:
  prompt_seconds
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from rgad_crosslingual_tts.audio import build_duration_filler, write_prompt_crop
from rgad_crosslingual_tts.constants import (
    DEFAULT_CHECKPOINT,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_STEP,
    DEFAULT_PROMPT_SECONDS,
    DEFAULT_SPEED,
    DEFAULT_T_SHIFT,
    DEFAULT_TOKENIZER,
)
from rgad_crosslingual_tts.infer import resolve_model_dir, resolve_zipvoice_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", required=True, type=Path)
    parser.add_argument("--res-dir", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--zipvoice-root", type=Path, default=None)
    parser.add_argument("--prompt-seconds", type=float, default=DEFAULT_PROMPT_SECONDS)
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--num-step", type=int, default=DEFAULT_NUM_STEP)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--t-shift", type=float, default=DEFAULT_T_SHIFT)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num-thread", type=int, default=1)
    parser.add_argument("--max-duration", type=float, default=100.0)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_dir, download=not args.no_download)
    zipvoice_root = resolve_zipvoice_root(args.zipvoice_root)
    args.res_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="rgad_tts_batch_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        zipvoice_tsv = tmpdir_path / "zipvoice_input.tsv"
        rows = []
        for index, line in enumerate(args.input_tsv.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) not in (3, 4):
                raise ValueError(f"{args.input_tsv}:{index + 1} expected 3 or 4 columns")
            wav_name, prompt_wav, text = parts[:3]
            seconds = float(parts[3]) if len(parts) == 4 else args.prompt_seconds
            cropped = tmpdir_path / "prompts" / f"{wav_name}.wav"
            actual_seconds = write_prompt_crop(prompt_wav, cropped, seconds=seconds)
            rows.append(f"{wav_name}\t{build_duration_filler(actual_seconds)}\t{cropped}\t{text}")
        zipvoice_tsv.write_text("\n".join(rows) + "\n", encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(zipvoice_root)
            if not env.get("PYTHONPATH")
            else str(zipvoice_root) + os.pathsep + env["PYTHONPATH"]
        )
        env["CUDA_VISIBLE_DEVICES"] = "" if str(args.gpu).lower() == "cpu" else str(args.gpu)
        cmd = [
            sys.executable,
            "-m",
            "zipvoice.bin.infer_zipvoice",
            "--model-name",
            "zipvoice",
            "--model-dir",
            str(model_dir),
            "--checkpoint-name",
            DEFAULT_CHECKPOINT,
            "--tokenizer",
            DEFAULT_TOKENIZER,
            "--test-list",
            str(zipvoice_tsv),
            "--res-dir",
            str(args.res_dir),
            "--num-step",
            str(args.num_step),
            "--speed",
            str(args.speed),
            "--guidance-scale",
            str(args.guidance_scale),
            "--t-shift",
            str(args.t_shift),
            "--num-thread",
            str(args.num_thread),
            "--max-duration",
            str(args.max_duration),
        ]
        subprocess.run(cmd, env=env, check=True)
    print(args.res_dir)


if __name__ == "__main__":
    main()
