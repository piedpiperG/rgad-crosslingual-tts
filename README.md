# RGAD Cross-Lingual TTS

RGAD Cross-Lingual TTS is a release package for the best current cross-lingual
Chinese TTS checkpoint from the RGAD-TTS project.

It converts a foreign-language prompt audio into Chinese speech while preserving
the prompt speaker's voice characteristics. The released model is a ZipVoice-style
flow-matching TTS student fine-tuned with reward-gated cross-lingual prompt-prefix
data.

- GitHub: <https://github.com/piedpiperG/rgad-crosslingual-tts>
- Model weights: <https://huggingface.co/isabeth/rgad-crosslingual-tts>
- Base architecture: ZipVoice, pinned to commit `2f7326fbfe999a3ad179e3f1af82a424d4a62819`

## What this model is for

Use this model when you have:

1. A short prompt audio from a speaker in any language.
2. Chinese text to synthesize.
3. A requirement for local, low-latency inference.

The recommended inference policy is:

- Crop the prompt audio to the first 6 seconds.
- Do not pass the foreign transcript as prompt text.
- Use a Chinese duration filler prompt text based on prompt duration.
- Use `--num-step 8 --speed 1.10`.

The wrapper scripts in this repo apply that policy automatically.

## Current recommended checkpoint

The default Hugging Face files correspond to:

- Source experiment: `stage16_crosslingual_hardcase_x90_4k_20260514`
- Checkpoint: `best-valid-loss.pt`
- Best validation iteration: 2500
- Tokenizer: `emilia`
- Sample rate: 24 kHz
- Default inference: `num_step=8`, `speed=1.10`

Internal fixed held-out cross-lingual evaluation:

| Set | Items | CER | SIM-o | UTMOS | RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| Podcast foreign prompt -> Chinese | 425 | 3.38% | 0.453 | 2.630 | 0.0564 |
| FLEURS foreign prompt -> Chinese | 946 | 8.60% | 0.512 | 3.244 | - |

The model is not claimed to be a public Chinese TTS SOTA model. Its main value is
local cross-lingual voice cloning with a compact ZipVoice student.

## Installation

Create a Python environment with CUDA-enabled PyTorch if you want GPU inference.
Python 3.10 is recommended.

```bash
git clone https://github.com/piedpiperG/rgad-crosslingual-tts.git
cd rgad-crosslingual-tts

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .

python scripts/setup_zipvoice.py
```

`setup_zipvoice.py` clones ZipVoice into `third_party/ZipVoice` and checks out the
commit used by this release. The inference scripts set `PYTHONPATH` automatically.

## Download model weights

```bash
python scripts/download_model.py --output-dir models/rgad-crosslingual-tts
```

The model directory must contain:

```text
best-valid-loss.pt
model.json
tokens.txt
run_config.json
train_summary.json
```

You can also skip this step. `scripts/infer.py` downloads to
`~/.cache/rgad-crosslingual-tts/model` automatically if no local model is found.

## Single-sentence inference

```bash
python scripts/infer.py \
  --model-dir models/rgad-crosslingual-tts \
  --prompt-wav /path/to/foreign_speaker.wav \
  --text "这是用外语说话人音色合成的中文语音。" \
  --output-wav outputs/demo.wav \
  --gpu 0
```

The script will:

1. Load and crop the prompt to the first 6 seconds.
2. Resample it to mono 24 kHz.
3. Build a Chinese duration filler, e.g. `嗯嗯嗯...。`.
4. Call ZipVoice with the released checkpoint.

Useful options:

```bash
--prompt-seconds 6
--speed 1.10
--num-step 8
--guidance-scale 1.0
--gpu cpu
```

## Batch inference

Create a TSV:

```text
utt001	/path/to/speaker_a.wav	这是第一条中文文本。
utt002	/path/to/speaker_b.wav	这是第二条中文文本。
```

Run:

```bash
python scripts/infer_batch.py \
  --model-dir models/rgad-crosslingual-tts \
  --input-tsv examples/batch.tsv \
  --res-dir outputs/batch \
  --gpu 0
```

Each output wav is saved as `{wav_name}.wav` under `--res-dir`.

## Fine-tuning

This repo includes the prefix fine-tuning recipe used by the RGAD-TTS cross-lingual
line. Your input JSONL should contain one row per training item:

```json
{"id":"sample_001","prompt_wav":"/path/to/foreign_prompt.wav","target_wav":"/path/to/chinese_target.wav","text":"中文目标文本。","prompt_language":"en","target_language":"zh-CN","speaker_id":"speaker_a"}
```

Build Lhotse manifests:

```bash
python scripts/prepare_prefix_manifest.py \
  --input-jsonl data/train.jsonl \
  --output-dir data/prefix_manifest \
  --split train \
  --prompt-text-policy duration_filler \
  --prompt-crop-seconds 6

python scripts/prepare_prefix_manifest.py \
  --input-jsonl data/dev.jsonl \
  --output-dir data/prefix_manifest \
  --split dev \
  --prompt-text-policy duration_filler \
  --prompt-crop-seconds 6
```

Fine-tune:

```bash
export PYTHONPATH="$PWD/third_party/ZipVoice:$PYTHONPATH"

python scripts/train_prefix.py \
  --train-manifest data/prefix_manifest/cuts_train.jsonl.gz \
  --dev-manifest data/prefix_manifest/cuts_dev.jsonl.gz \
  --model-config models/rgad-crosslingual-tts/model.json \
  --checkpoint models/rgad-crosslingual-tts/best-valid-loss.pt \
  --token-file models/rgad-crosslingual-tts/tokens.txt \
  --exp-dir runs/my_finetune \
  --num-iters 4000 \
  --base-lr 6e-6 \
  --condition-drop-ratio 0.2
```

Training writes `best-valid-loss.pt`, periodic checkpoints, `model.json`,
`tokens.txt`, `run_config.json`, and `train_summary.json` to `--exp-dir`.

## Python API

```python
from rgad_crosslingual_tts import synthesize

synthesize(
    prompt_wav="/path/to/foreign_prompt.wav",
    text="这是一个中文测试。",
    output_wav="outputs/api_demo.wav",
    model_dir="models/rgad-crosslingual-tts",
    gpu="0",
)
```

## Implementation notes

The released checkpoint uses the original ZipVoice model class. This repository
adds release-quality wrappers and training utilities around it. The current
cross-lingual improvement comes from data construction and prompt policy:

- Prompt-prefix fine-tuning.
- Cross-lingual prompt normalization with duration filler.
- Prompt crop to 6 seconds.
- Reward-gated hardcase fine-tuning.

Future architecture work should focus on:

1. Explicit speaker-only prompt mode.
2. Learned duration aligner instead of token-ratio duration.
3. Speaker side-channel and speaker consistency loss.

## License and attribution

Code in this release is Apache-2.0. ZipVoice is also Apache-2.0 and is used as
the base architecture and inference engine. The checkpoint is released for
research and application prototyping; verify rights for any training data or
speaker prompt audio you use.
