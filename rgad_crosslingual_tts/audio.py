"""Audio helpers for RGAD cross-lingual inference and data preparation."""

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio

from rgad_crosslingual_tts.constants import (
    DEFAULT_FILLER_CHARS_PER_SECOND,
    DEFAULT_PROMPT_SECONDS,
    DEFAULT_SAMPLE_RATE,
)


def build_duration_filler(
    seconds: float = DEFAULT_PROMPT_SECONDS,
    *,
    filler_char: str = "嗯",
    chars_per_second: float = DEFAULT_FILLER_CHARS_PER_SECOND,
    min_chars: int = 1,
    max_chars: int = 80,
) -> str:
    """Build the Chinese duration-filler prompt text used by the released model."""

    count = round(max(0.0, seconds) * chars_per_second)
    count = max(min_chars, min(max_chars, count))
    return filler_char * count + "。"


def load_mono_resampled(path: str | Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> torch.Tensor:
    """Load an audio file as mono waveform at the requested sample rate."""

    wav, source_rate = torchaudio.load(str(path))
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        wav = torchaudio.functional.resample(wav, source_rate, sample_rate)
    return wav


def write_prompt_crop(
    prompt_wav: str | Path,
    output_wav: str | Path,
    *,
    seconds: float = DEFAULT_PROMPT_SECONDS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> float:
    """Crop the prompt to the first N seconds, convert to mono 24 kHz, and save it."""

    wav = load_mono_resampled(prompt_wav, sample_rate=sample_rate)
    max_frames = int(round(seconds * sample_rate))
    if max_frames > 0:
        wav = wav[:, :max_frames]
    output_path = Path(output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), wav, sample_rate=sample_rate)
    return wav.shape[-1] / sample_rate


def concat_prompt_target(
    prompt_wav: str | Path,
    target_wav: str | Path,
    output_wav: str | Path,
    *,
    prompt_crop_seconds: float | None = DEFAULT_PROMPT_SECONDS,
    prompt_trailing_silence_seconds: float = 0.2,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> dict[str, float]:
    """Create prompt + silence + target audio for prefix fine-tuning manifests."""

    prompt = load_mono_resampled(prompt_wav, sample_rate=sample_rate)
    if prompt_crop_seconds is not None:
        prompt = prompt[:, : int(round(prompt_crop_seconds * sample_rate))]
    target = load_mono_resampled(target_wav, sample_rate=sample_rate)
    silence = torch.zeros(1, int(round(prompt_trailing_silence_seconds * sample_rate)))
    concat = torch.cat([prompt, silence, target], dim=1)
    output_path = Path(output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), concat, sample_rate=sample_rate, encoding="PCM_S", bits_per_sample=16)
    return {
        "prompt_audio_duration_seconds": prompt.shape[-1] / sample_rate,
        "prompt_condition_duration_seconds": (prompt.shape[-1] + silence.shape[-1]) / sample_rate,
        "target_duration_seconds": target.shape[-1] / sample_rate,
        "concat_duration_seconds": concat.shape[-1] / sample_rate,
    }
