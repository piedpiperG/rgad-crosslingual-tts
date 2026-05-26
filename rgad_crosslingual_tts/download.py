"""Download the released model from Hugging Face."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

from rgad_crosslingual_tts.constants import DEFAULT_HF_REPO_ID, DEFAULT_MODEL_CACHE


def download_model(
    output_dir: str | Path | None = None,
    *,
    repo_id: str = DEFAULT_HF_REPO_ID,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Download model files and return the local model directory."""

    local_dir = Path(output_dir or os.environ.get("RGAD_TTS_MODEL_DIR") or DEFAULT_MODEL_CACHE)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        local_dir=str(local_dir),
        token=token,
        allow_patterns=[
            "best-valid-loss.pt",
            "model.json",
            "tokens.txt",
            "run_config.json",
            "train_summary.json",
            "README.md",
        ],
    )
    return local_dir
