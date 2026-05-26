# Training Notes

This repository provides a compact fine-tuning path, not the full RGAD-TTS
teacher-candidate production pipeline.

## Data format

Prepare JSONL rows with prompt audio, target audio, and target text:

```json
{"id":"sample_001","prompt_wav":"/data/prompt.wav","target_wav":"/data/target.wav","text":"中文目标文本。","prompt_language":"en","target_language":"zh-CN","speaker_id":"spk1"}
```

The manifest builder concatenates:

```text
prompt audio + 0.2s silence + target audio
```

The trainer masks only the target region and computes flow-matching loss there.
The prompt region remains visible as acoustic condition.

## Prompt text policy

The recommended policy is `duration_filler`. It ignores the foreign prompt
transcript and creates Chinese filler text proportional to prompt duration:

```python
"嗯" * round(prompt_seconds * 4) + "。"
```

This is the policy used by the released Stage16 checkpoint.

## Minimal workflow

```bash
python scripts/prepare_prefix_manifest.py \
  --input-jsonl data/train.jsonl \
  --output-dir data/prefix_manifest \
  --split train

python scripts/prepare_prefix_manifest.py \
  --input-jsonl data/dev.jsonl \
  --output-dir data/prefix_manifest \
  --split dev

python scripts/train_prefix.py \
  --train-manifest data/prefix_manifest/cuts_train.jsonl.gz \
  --dev-manifest data/prefix_manifest/cuts_dev.jsonl.gz \
  --model-config models/rgad-crosslingual-tts/model.json \
  --checkpoint models/rgad-crosslingual-tts/best-valid-loss.pt \
  --token-file models/rgad-crosslingual-tts/tokens.txt \
  --exp-dir runs/my_finetune
```

## Recommended starting hyperparameters

```text
num_iters: 4000
base_lr: 6e-6
condition_drop_ratio: 0.2
max_duration: 120
num_buckets: 20
prompt_crop_seconds: 6
```

Use a small dev set that matches your deployment domain. Select checkpoints by
both content error and speaker similarity, not validation loss alone.
