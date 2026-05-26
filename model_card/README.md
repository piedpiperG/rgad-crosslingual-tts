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

这是 RGAD-TTS 项目当前最推荐的跨语言中文 TTS checkpoint。

它的用途是：**输入一段外语说话人的 prompt audio，保留该说话人的音色，用中文文本合成中文语音**。

配套 GitHub 使用仓库：

<https://github.com/piedpiperG/rgad-crosslingual-tts>

## 模型信息

- 底座架构：ZipVoice
- 原始实验：`stage16_crosslingual_hardcase_x90_4k_20260514`
- checkpoint：`best-valid-loss.pt`
- tokenizer：`emilia`
- 输出采样率：24 kHz
- 推荐推理参数：`num_step=8`，`speed=1.10`

## 适用场景

输入：

1. 一段说话人 prompt audio，通常可以是非中文语音。
2. 一段中文目标文本。

输出：

使用 prompt 说话人音色合成出的中文语音。

推荐推理策略：

- prompt audio 裁剪前 6 秒。
- 不使用外语 transcript 作为 prompt text。
- 根据 prompt 时长构造中文 duration filler，例如 `嗯嗯嗯...。`。
- 使用 `speed=1.10`。

GitHub 仓库中的 `scripts/infer.py` 已经自动封装这些步骤。

## 快速开始

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
  --output-wav outputs/demo.wav \
  --gpu 0
```

## 文件说明

本模型仓库包含：

- `best-valid-loss.pt`：推荐 checkpoint。
- `model.json`：ZipVoice 模型结构配置。
- `tokens.txt`：Emilia tokenizer 词表。
- `run_config.json`：训练配置。
- `train_summary.json`：训练摘要。

## 内部评测

| 评测集 | 样本数 | CER ↓ | SIM-o ↑ | UTMOS ↑ | RTF ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Podcast 外语 prompt -> 中文 | 425 | 3.38% | 0.453 | 2.630 | 0.0564 |
| FLEURS 外语 prompt -> 中文 | 946 | 8.60% | 0.512 | 3.244 | - |

这些是项目内部固定评测结果，不应理解为公开中文 TTS SOTA 声明。

## 模型特点

- 面向外语 prompt 到中文 TTS 的跨语言克隆。
- 使用 ZipVoice-style flow-matching student，本地推理成本较低。
- 推荐使用 duration filler 避免外语 transcript 干扰中文 target。
- 适合在该仓库基础上继续做 speaker-only prompt mode、duration aligner、speaker loss 等架构改进。

## 局限性

- 主要面向中文目标文本。
- prompt audio 如果噪声大、多人说话或过长，音色保持和内容稳定性会下降。
- 当前 checkpoint 仍使用原 ZipVoice duration 机制，长句韵律和停顿仍有改进空间。

## 许可

模型和配套代码按 Apache-2.0 发布。ZipVoice 底座同样为 Apache-2.0。
使用任何 prompt audio 或训练数据时，请自行确认数据和声音授权。
