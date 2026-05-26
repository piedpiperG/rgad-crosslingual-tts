"""Inference wrapper around ZipVoice for the released RGAD model."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from rgad_crosslingual_tts.audio import build_duration_filler, write_prompt_crop
from rgad_crosslingual_tts.constants import (
    DEFAULT_CHECKPOINT,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_MODEL_CACHE,
    DEFAULT_NUM_STEP,
    DEFAULT_PROMPT_SECONDS,
    DEFAULT_SPEED,
    DEFAULT_T_SHIFT,
    DEFAULT_TOKENIZER,
    DEFAULT_ZIPVOICE_ROOT,
)
from rgad_crosslingual_tts.download import download_model


def resolve_model_dir(model_dir: str | Path | None = None, *, download: bool = True) -> Path:
    """Resolve a local model directory, downloading from Hugging Face if needed."""

    resolved = Path(model_dir or os.environ.get("RGAD_TTS_MODEL_DIR") or DEFAULT_MODEL_CACHE)
    required = [resolved / DEFAULT_CHECKPOINT, resolved / "model.json", resolved / "tokens.txt"]
    if all(path.is_file() for path in required):
        return resolved
    if not download:
        missing = ", ".join(str(path) for path in required if not path.is_file())
        raise FileNotFoundError(f"Missing model files: {missing}")
    return download_model(resolved)


def resolve_zipvoice_root(zipvoice_root: str | Path | None = None) -> Path:
    """Resolve the ZipVoice source tree used through PYTHONPATH."""

    root = Path(zipvoice_root or os.environ.get("ZIPVOICE_ROOT") or DEFAULT_ZIPVOICE_ROOT)
    if not (root / "zipvoice" / "bin" / "infer_zipvoice.py").is_file():
        raise FileNotFoundError(
            f"ZipVoice was not found at {root}. Run: python scripts/setup_zipvoice.py"
        )
    return root


def synthesize(
    *,
    prompt_wav: str | Path,
    text: str,
    output_wav: str | Path,
    model_dir: str | Path | None = None,
    checkpoint_name: str = DEFAULT_CHECKPOINT,
    zipvoice_root: str | Path | None = None,
    prompt_seconds: float = DEFAULT_PROMPT_SECONDS,
    prompt_text: str | None = None,
    speed: float = DEFAULT_SPEED,
    num_step: int = DEFAULT_NUM_STEP,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    t_shift: float = DEFAULT_T_SHIFT,
    gpu: str | None = "0",
    num_thread: int = 1,
    python: str | Path | None = None,
    download: bool = True,
    vocoder_path: str | Path | None = None,
) -> Path:
    """Generate Chinese speech with the released cross-lingual checkpoint."""

    model_path = resolve_model_dir(model_dir, download=download)
    zipvoice_path = resolve_zipvoice_root(zipvoice_root)
    output_path = Path(output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="rgad_tts_") as tmpdir:
        cropped_prompt = Path(tmpdir) / "prompt_first_seconds.wav"
        actual_seconds = write_prompt_crop(prompt_wav, cropped_prompt, seconds=prompt_seconds)
        filler = prompt_text or build_duration_filler(actual_seconds)

        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(zipvoice_path)
            if not env.get("PYTHONPATH")
            else str(zipvoice_path) + os.pathsep + env["PYTHONPATH"]
        )
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = "" if str(gpu).lower() == "cpu" else str(gpu)

        cmd = [
            str(python or sys.executable),
            "-m",
            "zipvoice.bin.infer_zipvoice",
            "--model-name",
            "zipvoice",
            "--model-dir",
            str(model_path),
            "--checkpoint-name",
            checkpoint_name,
            "--tokenizer",
            DEFAULT_TOKENIZER,
            "--prompt-wav",
            str(cropped_prompt),
            "--prompt-text",
            filler,
            "--text",
            text,
            "--res-wav-path",
            str(output_path),
            "--num-step",
            str(num_step),
            "--speed",
            str(speed),
            "--guidance-scale",
            str(guidance_scale),
            "--t-shift",
            str(t_shift),
            "--num-thread",
            str(num_thread),
        ]
        if vocoder_path is not None:
            cmd.extend(["--vocoder-path", str(vocoder_path)])
        subprocess.run(cmd, env=env, check=True)
    return output_path
