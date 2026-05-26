# RGAD Cross-Lingual TTS

这是 RGAD-TTS 项目当前最推荐的跨语言中文 TTS 推理发行版。

它的用途是：**输入一段外语说话人的 prompt audio，保留该说话人的音色，用中文文本合成中文语音**。
模型主体是 ZipVoice-style flow-matching TTS student，权重来自 RGAD-TTS 的 Stage16
cross-lingual hardcase fine-tuning。

- GitHub 仓库：<https://github.com/piedpiperG/rgad-crosslingual-tts>
- HuggingFace 权重：<https://huggingface.co/isabeth/rgad-crosslingual-tts>
- 底座架构：ZipVoice，固定 commit `2f7326fbfe999a3ad179e3f1af82a424d4a62819`

## 模型适合做什么

适合以下场景：

1. 你有一段外语 prompt audio，例如英语、日语、韩语、播客片段等。
2. 你希望用这个说话人的音色合成中文文本。
3. 你希望在本地低延迟运行，而不是调用大型在线 TTS teacher。

推荐推理策略已经封装在本仓库脚本里：

- prompt audio 默认裁剪前 6 秒。
- 不把外语 transcript 传给模型。
- 按 prompt 时长自动构造中文 duration filler，例如 `嗯嗯嗯...。`。
- 默认使用 `--num-step 8 --speed 1.10`。

## 当前推荐权重

默认下载的 HuggingFace 权重对应：

- 原始实验：`stage16_crosslingual_hardcase_x90_4k_20260514`
- checkpoint：`best-valid-loss.pt`
- best valid iter：2500
- tokenizer：`emilia`
- 输出采样率：24 kHz
- 默认推理参数：`num_step=8`，`speed=1.10`

## 跨语言 TTS 评测表现

下面结果整理自原 RGAD-TTS 项目的 `docs/paper_assets/stage21_main_text_20260520` 主表。
所有系统使用相同 prompt audio、目标文本、ASR、SIM-o、UTMOS 和 RTF 评测协议；目标语言为中文，因此主要看 CER。

### FLEURS 公共跨语言基准

| 系统 | 类型 | 样本数 | CER ↓ | SIM-o ↑ | UTMOS ↑ | RTF ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original compact student | ZipVoice-style compact student | 946 | 51.31% | 0.551 | 2.988 | 0.0548 |
| F5-TTS | 开源小/中型 TTS baseline | 946 | 21.22% | 0.526 | 2.699 | 0.1364 |
| IndexTTS2 teacher | 大 teacher baseline | 946 | 3.68% | 0.667 | 2.979 | 0.9580 |
| Fish Audio S2 teacher | 大 teacher baseline | 946 | 7.25% | 0.642 | 3.516 | 0.5104 |
| CosyVoice3 teacher | 大 teacher baseline | 946 | 20.80% | 0.674 | 3.338 | 0.5705 |
| **RGAD-TTS release** | 本仓库发行权重 | 946 | **13.70%** | 0.512 | **3.244** | **0.0565** |
| Reference target audio | 目标音频参考 | 946 | 4.15% | 0.066 | 2.727 | - |

在 FLEURS 上，RGAD-TTS 将原始 compact student 的 CER 从 51.31% 降到 13.70%，并低于 F5-TTS 的 21.22%。
IndexTTS2 和 Fish Audio S2 teacher 在 CER 上仍更强，但 RTF 分别为 0.9580 和 0.5104；RGAD-TTS 的定位是把跨语言克隆能力压缩到可本地快速推理的 student 中。

### Podcast held-out 真实跨语言配音基准

| 系统 | 类型 | 样本数 | CER ↓ | SIM-o ↑ | UTMOS ↑ | RTF ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Original compact student | ZipVoice-style compact student | 425 | 49.10% | 0.488 | 2.662 | 0.0672 |
| F5-TTS | 开源小/中型 TTS baseline | 425 | 54.81% | 0.426 | 1.939 | 0.0626 |
| IndexTTS2 teacher | 大 teacher baseline | 425 | 2.87% | 0.509 | 2.546 | 1.4839 |
| Fish Audio S2 teacher | 大 teacher baseline | 425 | 169.91% | 0.605 | 3.631 | 0.5035 |
| CosyVoice3 teacher | 大 teacher baseline | 425 | 105.95% | 0.563 | 3.335 | 0.4570 |
| **RGAD-TTS release** | 本仓库发行权重 | 425 | **3.38%** | 0.453 | 2.630 | **0.0564** |
| Podcast target audio | 目标音频参考 | 425 | 2.67% | 0.501 | 2.535 | - |

Podcast held-out 更接近真实外语视频/播客配音场景。RGAD-TTS 的 CER 为 3.38%，接近 IndexTTS2 teacher 的 2.87% 和目标音频参考的 2.67%，但 RTF 约为 IndexTTS2 的 1/26。
在该基准下，F5-TTS 和部分大 teacher 会出现明显内容错误，说明大模型并不总是在跨语言 prompt 到中文 target 的流水线里稳定。

### 核心消融

| 配置 | 样本数 | CER ↓ | SIM-o ↑ | UTMOS ↑ | RTF ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Full RGAD-TTS** | 946 | **13.70%** | 0.512 | 3.244 | 0.0565 |
| w/o reward gate | 946 | 46.96% | 0.528 | 2.743 | 0.0549 |
| w/o prompt normalization | 946 | 61.73% | 0.493 | 3.029 | 0.0564 |
| single-teacher distillation | 946 | 28.47% | 0.494 | 3.070 | 0.0567 |

这组消融说明：当前发行权重的提升主要来自 reward-gated acoustic supervision、duration filler/prompt normalization 和多 teacher 目标筛选，而不是单纯增加跨语言 paired data。

## 视频 Demo

以下 demo 来自 `D:\C-data\rgad_stage16_compare_20260515_171437` 的前三个 case，已随仓库放在 `assets/demo_videos/`。
每个 case 同时给出源视频、历史 IndexTTS2 输出和本发行版 RGAD-TTS 输出，方便直接对比。

| Case | 目标中文首句 | 源视频 | IndexTTS2 输出 | RGAD-TTS 输出 | 字幕 |
| --- | --- | --- | --- | --- | --- |
| 01 | 我们正遭遇此生前所未有的最大危机。 | [source](assets/demo_videos/source_videos/01_run_20260513_195502_source.mp4) | [IndexTTS2](assets/demo_videos/indextts2_outputs/01_run_20260513_195502_indextts2.mp4) | [RGAD-TTS](assets/demo_videos/rgad_outputs/01_run_20260513_195502_rgad.mp4) | [SRT](assets/demo_videos/subtitles/01_run_20260513_195502.srt) |
| 02 | 这套超值组合内含六件 T 恤，大家快看，款式多漂亮！ | [source](assets/demo_videos/source_videos/02_run_20260513_183602_source.mp4) | [IndexTTS2](assets/demo_videos/indextts2_outputs/02_run_20260513_183602_indextts2.mp4) | [RGAD-TTS](assets/demo_videos/rgad_outputs/02_run_20260513_183602_rgad.mp4) | [SRT](assets/demo_videos/subtitles/02_run_20260513_183602.srt) |
| 03 | 我刚和马特-加明碰过面，他指出，到 两千零二十六 年，可用的 GPU 算力将几乎归零。 | [source](assets/demo_videos/source_videos/03_run_20260511_114936_source.mp4) | [IndexTTS2](assets/demo_videos/indextts2_outputs/03_run_20260511_114936_indextts2.mp4) | [RGAD-TTS](assets/demo_videos/rgad_outputs/03_run_20260511_114936_rgad.mp4) | [SRT](assets/demo_videos/subtitles/03_run_20260511_114936.srt) |

注意：这个模型不主张是公开中文 TTS SOTA。它的主要价值是把跨语言克隆能力压缩到一个可本地部署、推理成本较低的 ZipVoice student 中。

## 安装

推荐 Python 3.10。GPU 推理需要 CUDA 版 PyTorch。

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

`setup_zipvoice.py` 会把 ZipVoice 克隆到 `third_party/ZipVoice`，并切到本发行版验证过的 commit。
推理脚本会自动设置 `PYTHONPATH`，通常不需要手动改环境变量。

## 下载模型权重

```bash
python scripts/download_model.py --output-dir models/rgad-crosslingual-tts
```

模型目录应包含：

```text
best-valid-loss.pt
model.json
tokens.txt
run_config.json
train_summary.json
```

如果不提前下载，`scripts/infer.py` 在找不到本地模型时会自动从 HuggingFace 下载到：

```text
~/.cache/rgad-crosslingual-tts/model
```

## 单句推理

```bash
python scripts/infer.py \
  --model-dir models/rgad-crosslingual-tts \
  --prompt-wav /path/to/foreign_speaker.wav \
  --text "这是用外语说话人音色合成的中文语音。" \
  --output-wav outputs/demo.wav \
  --gpu 0
```

脚本内部会自动完成：

1. 读取 prompt audio。
2. 裁剪到前 6 秒。
3. 转成 mono 24 kHz。
4. 按实际 prompt 时长构造中文 duration filler。
5. 调用 ZipVoice 生成中文语音。

常用参数：

```bash
--prompt-seconds 6
--speed 1.10
--num-step 8
--guidance-scale 1.0
--gpu cpu
```

## 批量推理

准备 TSV 文件，每行三列：

```text
utt001	/path/to/speaker_a.wav	这是第一条中文文本。
utt002	/path/to/speaker_b.wav	这是第二条中文文本。
```

运行：

```bash
python scripts/infer_batch.py \
  --model-dir models/rgad-crosslingual-tts \
  --input-tsv examples/batch.tsv \
  --res-dir outputs/batch \
  --gpu 0
```

输出文件会保存为：

```text
outputs/batch/{wav_name}.wav
```

## Python 调用

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

## 继续训练 / 微调

本仓库提供一个轻量 prefix fine-tuning 入口，复用当前跨语言训练链路。你的训练 JSONL 每行格式如下：

```json
{"id":"sample_001","prompt_wav":"/path/to/foreign_prompt.wav","target_wav":"/path/to/chinese_target.wav","text":"中文目标文本。","prompt_language":"en","target_language":"zh-CN","speaker_id":"speaker_a"}
```

构建 Lhotse manifest：

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

开始微调：

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

训练输出包括：

```text
best-valid-loss.pt
checkpoint-*.pt
model.json
tokens.txt
run_config.json
train_summary.json
train.log
```

更详细的训练说明见 [docs/TRAINING.md](docs/TRAINING.md)。

## 实现说明

当前发行版没有修改 ZipVoice 模型类本身，而是在 ZipVoice 外层封装了：

- 模型权重下载。
- 推荐跨语言 prompt 预处理。
- 单句和批量推理。
- prompt-prefix fine-tuning 数据准备。
- target 区域 flow-matching fine-tuning 脚本。

当前模型效果主要来自：

- prompt-prefix 跨语言训练。
- duration filler prompt normalization。
- prompt crop 6 秒。
- reward-gated hardcase fine-tuning。

后续建议优先改进的架构方向：

1. 显式 speaker-only prompt mode。
2. learned duration aligner，替换 token-ratio duration。
3. speaker side-channel 和 speaker consistency loss。

## 局限性

- 目标文本主要面向中文。
- 音色保持依赖 prompt audio 质量。
- 很长、噪声很大、多人说话的 prompt 可能不稳定。
- 当前模型仍沿用 ZipVoice 原始 duration 机制，复杂长句的停顿和韵律仍有提升空间。

## 许可和引用

本仓库代码使用 Apache-2.0。ZipVoice 也使用 Apache-2.0，作为本项目的底座架构和推理引擎：

<https://github.com/k2-fsa/ZipVoice>

使用任何说话人 prompt audio 或训练数据时，请自行确认数据和声音授权。
