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

## 配套数据集

10 小时跨语言 TTS 数据集：

<https://huggingface.co/datasets/isabeth/rgad-crosslingual-tts-10h>

数据集包含 `train.jsonl`、`dev.jsonl`、`metadata.csv`、`audio/prompts/*.wav`
和 `audio/targets/*.wav`。可配合 GitHub 仓库中的 `scripts/prepare_prefix_manifest.py`
构建 prefix fine-tuning manifest。JSONL 每行包含
`id`、`prompt_wav`、`target_wav`、`text`、`prompt_language`、`target_language`
和 `speaker_id`，音频路径相对数据集根目录。

## 跨语言 TTS 评测

下面结果整理自原 RGAD-TTS 项目的 `docs/paper_assets/stage21_main_text_20260520` 主表。
所有系统使用相同 prompt audio、目标文本、ASR、SIM-o、UTMOS 和 RTF 评测协议；目标语言为中文，因此主要看 CER。

### FLEURS 公共跨语言基准

| 系统 | 样本数 | CER ↓ | SIM-o ↑ | UTMOS ↑ | RTF ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original compact model | 946 | 51.31% | 0.551 | 2.988 | 0.0548 |
| F5-TTS | 946 | 21.22% | 0.526 | 2.699 | 0.1364 |
| IndexTTS2 | 946 | 3.68% | 0.667 | 2.979 | 0.9580 |
| Fish Audio S2 | 946 | 7.25% | 0.642 | 3.516 | 0.5104 |
| CosyVoice3 | 946 | 20.80% | 0.674 | 3.338 | 0.5705 |
| **RGAD-TTS release** | 946 | **13.70%** | 0.512 | **3.244** | **0.0565** |
| Reference target audio | 946 | 4.15% | 0.066 | 2.727 | - |

### Podcast held-out 真实跨语言配音基准

| 系统 | 样本数 | CER ↓ | SIM-o ↑ | UTMOS ↑ | RTF ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original compact model | 425 | 49.10% | 0.488 | 2.662 | 0.0672 |
| F5-TTS | 425 | 54.81% | 0.426 | 1.939 | 0.0626 |
| IndexTTS2 | 425 | 2.87% | 0.509 | 2.546 | 1.4839 |
| Fish Audio S2 | 425 | 169.91% | 0.605 | 3.631 | 0.5035 |
| CosyVoice3 | 425 | 105.95% | 0.563 | 3.335 | 0.4570 |
| **RGAD-TTS release** | 425 | **3.38%** | 0.453 | 2.630 | **0.0564** |
| Podcast target audio | 425 | 2.67% | 0.501 | 2.535 | - |

RGAD-TTS 在 FLEURS 上明显超过 F5-TTS 和原始 compact model；在 Podcast held-out 上，CER 接近 IndexTTS2 和目标音频参考，但 RTF 约为 IndexTTS2 的 1/26。
IndexTTS2 和 Fish Audio S2 在部分指标上仍更强，因此这里不声明公开中文 TTS SOTA；本模型的核心价值是提供一个本地可部署、推理成本较低的跨语言中文 TTS 模型。

## 模型特点

- 面向外语 prompt 到中文 TTS 的跨语言克隆。
- 使用 ZipVoice-style flow-matching compact model，本地推理成本较低。
- 推荐使用 duration filler 避免外语 transcript 干扰中文 target。
- 适合在该仓库基础上继续做 speaker-only prompt mode、duration aligner、speaker loss 等架构改进。

## 局限性

- 主要面向中文目标文本。
- prompt audio 如果噪声大、多人说话或过长，音色保持和内容稳定性会下降。
- 当前 checkpoint 仍使用原 ZipVoice duration 机制，长句韵律和停顿仍有改进空间。

## 许可

模型和配套代码按 Apache-2.0 发布。ZipVoice 底座同样为 Apache-2.0。
使用任何 prompt audio 或训练数据时，请自行确认数据和声音授权。
