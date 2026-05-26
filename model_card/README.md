---
license: apache-2.0
language:
- zh
tags:
- text-to-speech
- voice-cloning
- cross-lingual-tts
- zipvoice
- flow-matching
library_name: pytorch
pipeline_tag: text-to-speech
---

# RGAD Cross-Lingual TTS

This repository contains the released checkpoint for RGAD Cross-Lingual TTS,
a ZipVoice-style flow-matching student for foreign-language prompt audio to
Chinese speech synthesis.

GitHub usage package:
<https://github.com/piedpiperG/rgad-crosslingual-tts>

## Model

- Base architecture: ZipVoice
- Source experiment: `stage16_crosslingual_hardcase_x90_4k_20260514`
- Checkpoint: `best-valid-loss.pt`
- Tokenizer: `emilia`
- Sample rate: 24 kHz
- Recommended inference: `num_step=8`, `speed=1.10`

## Intended use

Input:

1. A short prompt audio from a speaker, usually in a non-Chinese language.
2. Chinese target text.

Output:

Chinese speech in the prompt speaker's voice style.

The recommended prompt policy is to crop the prompt audio to the first 6 seconds
and use a Chinese duration filler as prompt text rather than the foreign transcript.

## Quick start

```bash
git clone https://github.com/piedpiperG/rgad-crosslingual-tts.git
cd rgad-crosslingual-tts
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/setup_zipvoice.py
python scripts/download_model.py --output-dir models/rgad-crosslingual-tts

python scripts/infer.py \
  --model-dir models/rgad-crosslingual-tts \
  --prompt-wav /path/to/foreign_speaker.wav \
  --text "这是用外语说话人音色合成的中文语音。" \
  --output-wav outputs/demo.wav
```

## Internal evaluation

| Set | Items | CER | SIM-o | UTMOS | RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| Podcast foreign prompt -> Chinese | 425 | 3.38% | 0.453 | 2.630 | 0.0564 |
| FLEURS foreign prompt -> Chinese | 946 | 8.60% | 0.512 | 3.244 | - |

These are internal evaluation numbers and should not be interpreted as a public
Chinese TTS SOTA claim.

## Files

- `best-valid-loss.pt`: released checkpoint
- `model.json`: ZipVoice model config
- `tokens.txt`: tokenizer vocabulary
- `run_config.json`: fine-tuning configuration
- `train_summary.json`: training summary

## Limitations

- The model is optimized for Chinese target text.
- Speaker similarity depends on prompt quality and recording conditions.
- The current architecture still uses ZipVoice's original duration mechanism;
  future work should add explicit speaker-only prompt mode and learned duration
  alignment.
