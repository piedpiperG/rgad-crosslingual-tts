"""Project constants."""

from __future__ import annotations

from pathlib import Path

DEFAULT_HF_REPO_ID = "isabeth/rgad-crosslingual-tts"
DEFAULT_CHECKPOINT = "best-valid-loss.pt"
DEFAULT_TOKENIZER = "emilia"
DEFAULT_PROMPT_SECONDS = 6.0
DEFAULT_FILLER_CHARS_PER_SECOND = 4.0
DEFAULT_NUM_STEP = 8
DEFAULT_SPEED = 1.10
DEFAULT_T_SHIFT = 0.5
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_SAMPLE_RATE = 24000
ZIPVOICE_COMMIT = "2f7326fbfe999a3ad179e3f1af82a424d4a62819"

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIPVOICE_ROOT = PACKAGE_ROOT / "third_party" / "ZipVoice"
DEFAULT_MODEL_CACHE = Path.home() / ".cache" / "rgad-crosslingual-tts" / "model"
