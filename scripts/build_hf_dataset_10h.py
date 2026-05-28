#!/usr/bin/env python3
"""Build a lightweight Hugging Face dataset directory for prefix TTS training."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def read_cuts(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as file_obj:
        for line in file_obj:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def hhmmss(seconds: float) -> str:
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "item"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def custom(cut: dict[str, Any]) -> dict[str, Any]:
    return dict(cut["supervisions"][0].get("custom") or {})


def target_duration(cut: dict[str, Any]) -> float:
    return float(custom(cut).get("target_duration_seconds") or 0.0)


def is_candidate(cut: dict[str, Any], target_language: str) -> bool:
    item = custom(cut)
    prompt_language = item.get("prompt_language")
    return (
        item.get("target_language") == target_language
        and prompt_language
        and prompt_language != target_language
        and item.get("prompt_audio")
        and item.get("target_audio")
        and target_duration(cut) > 0
        and Path(str(item["prompt_audio"])).is_file()
        and Path(str(item["target_audio"])).is_file()
    )


def select_rows(cuts: list[dict[str, Any]], target_seconds: float, target_language: str) -> list[dict[str, Any]]:
    candidates = [cut for cut in cuts if is_candidate(cut, target_language)]
    selected: list[dict[str, Any]] = []
    total = 0.0
    seen_targets: set[str] = set()

    for cut in candidates:
        target_audio = str(custom(cut)["target_audio"])
        if target_audio in seen_targets:
            continue
        selected.append(cut)
        seen_targets.add(target_audio)
        total += target_duration(cut)
        if total >= target_seconds:
            return selected

    for cut in candidates:
        if total >= target_seconds:
            break
        selected.append(cut)
        total += target_duration(cut)

    if total < target_seconds:
        raise RuntimeError(
            f"only found {hhmmss(total)} target audio, below requested {hhmmss(target_seconds)}"
        )
    return selected


def split_rows(rows: list[dict[str, Any]], dev_target_seconds: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dev: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    dev_total = 0.0
    for index, row in enumerate(rows):
        if index % 17 == 0 and dev_total < dev_target_seconds:
            dev.append(row)
            dev_total += float(row["target_duration_seconds"])
        else:
            train.append(row)
    return train, dev


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audio" / "prompts").mkdir(parents=True, exist_ok=True)
    (output_dir / "audio" / "targets").mkdir(parents=True, exist_ok=True)

    selected_cuts = select_rows(
        read_cuts(args.source_manifest),
        target_seconds=args.target_hours * 3600.0,
        target_language=args.target_language,
    )
    rows: list[dict[str, Any]] = []
    audio_paths: dict[tuple[str, str], Path] = {}
    used_audio_paths: dict[Path, str] = {}
    row_id_counts: Counter[str] = Counter()
    target_path_counts: Counter[str] = Counter()

    for index, cut in enumerate(selected_cuts):
        item = custom(cut)
        base_id = slug(str(cut.get("id") or f"sample_{index:06d}"))
        row_occurrence = row_id_counts[base_id]
        row_id_counts[base_id] += 1
        sample_id = base_id if row_occurrence == 0 else f"{base_id}__repeat{row_occurrence:04d}"
        prompt_src = Path(str(item["prompt_audio"]))
        target_src = Path(str(item["target_audio"]))
        prompt_rel = materialize_audio(
            prompt_src,
            output_dir=output_dir,
            subdir="prompts",
            base_id=base_id,
            audio_paths=audio_paths,
            used_audio_paths=used_audio_paths,
        )
        target_rel = materialize_audio(
            target_src,
            output_dir=output_dir,
            subdir="targets",
            base_id=base_id,
            audio_paths=audio_paths,
            used_audio_paths=used_audio_paths,
        )
        target_path_counts[str(target_src)] += 1
        rows.append(
            {
                "id": sample_id,
                "prompt_wav": prompt_rel.as_posix(),
                "target_wav": target_rel.as_posix(),
                "text": " ".join(str(item.get("target_text") or cut["supervisions"][0]["text"]).split()),
                "prompt_language": str(item.get("prompt_language") or "und"),
                "target_language": str(item.get("target_language") or args.target_language),
                "speaker_id": str(cut["supervisions"][0].get("speaker") or sample_id),
                "prompt_duration_seconds": float(item.get("prompt_audio_duration_seconds") or 0.0),
                "target_duration_seconds": float(item.get("target_duration_seconds") or 0.0),
                "prompt_sha256": file_sha256(output_dir / prompt_rel),
                "target_sha256": file_sha256(output_dir / target_rel),
            }
        )

    dev_target_seconds = args.dev_minutes * 60.0
    train_rows, dev_rows = split_rows(rows, dev_target_seconds)
    for split, split_rows_ in (("train", train_rows), ("dev", dev_rows)):
        write_jsonl(output_dir / f"{split}.jsonl", split_rows_)

    with (output_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()) + ["split"])
        writer.writeheader()
        dev_ids = {row["id"] for row in dev_rows}
        for row in rows:
            writer.writerow({**row, "split": "dev" if row["id"] in dev_ids else "train"})

    summary = {
        "dataset_name": args.dataset_name,
        "num_rows": len(rows),
        "num_train_rows": len(train_rows),
        "num_dev_rows": len(dev_rows),
        "target_duration_seconds": round(sum(row["target_duration_seconds"] for row in rows), 3),
        "target_duration_hhmmss": hhmmss(sum(row["target_duration_seconds"] for row in rows)),
        "train_target_duration_seconds": round(sum(row["target_duration_seconds"] for row in train_rows), 3),
        "train_target_duration_hhmmss": hhmmss(
            sum(row["target_duration_seconds"] for row in train_rows)
        ),
        "dev_target_duration_seconds": round(sum(row["target_duration_seconds"] for row in dev_rows), 3),
        "dev_target_duration_hhmmss": hhmmss(sum(row["target_duration_seconds"] for row in dev_rows)),
        "prompt_duration_seconds": round(sum(row["prompt_duration_seconds"] for row in rows), 3),
        "prompt_duration_hhmmss": hhmmss(sum(row["prompt_duration_seconds"] for row in rows)),
        "target_language_counts": dict(Counter(row["target_language"] for row in rows)),
        "prompt_language_counts": dict(Counter(row["prompt_language"] for row in rows)),
        "unique_source_target_audio": len(target_path_counts),
        "repeated_source_target_rows": sum(max(0, count - 1) for count in target_path_counts.values()),
        "files": ["README.md", "metadata.csv", "train.jsonl", "dev.jsonl", "audio/prompts/*.wav", "audio/targets/*.wav"],
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(dataset_card(summary), encoding="utf-8")
    return summary


def materialize_audio(
    source: Path,
    *,
    output_dir: Path,
    subdir: str,
    base_id: str,
    audio_paths: dict[tuple[str, str], Path],
    used_audio_paths: dict[Path, str],
) -> Path:
    source_key = (subdir, str(source.resolve()))
    if source_key in audio_paths:
        return audio_paths[source_key]

    rel_path = Path("audio") / subdir / f"{base_id}.wav"
    owner = used_audio_paths.get(rel_path)
    if owner is not None and owner != source_key[1]:
        rel_path = Path("audio") / subdir / f"{base_id}__{file_sha256(source)[:10]}.wav"

    used_audio_paths[rel_path] = source_key[1]
    audio_paths[source_key] = rel_path
    shutil.copy2(source, output_dir / rel_path)
    return rel_path


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def dataset_card(summary: dict[str, Any]) -> str:
    return f"""---
license: apache-2.0
language:
- zh
task_categories:
- text-to-speech
tags:
- cross-lingual-tts
- voice-cloning
- prompt-audio
pretty_name: RGAD Cross-Lingual TTS 10h
---

# RGAD Cross-Lingual TTS 10h

This is a 10-hour cross-lingual TTS dataset for prompt-conditioned Chinese TTS fine-tuning.

## Format

The dataset contains:

- `train.jsonl`
- `dev.jsonl`
- `metadata.csv`
- `audio/prompts/*.wav`
- `audio/targets/*.wav`

Each JSONL row has this format:

```json
{{"id":"sample_000001","prompt_wav":"audio/prompts/sample_000001.wav","target_wav":"audio/targets/sample_000001.wav","text":"中文目标文本。","prompt_language":"en-US","target_language":"zh-CN","speaker_id":"speaker_001"}}
```

Summary:

- rows: {summary["num_rows"]}
- target audio: {summary["target_duration_hhmmss"]}
- prompt audio: {summary["prompt_duration_hhmmss"]}
- train target audio: {summary["train_target_duration_hhmmss"]}
- dev target audio: {summary["dev_target_duration_hhmmss"]}

## Use With `rgad-crosslingual-tts`

```bash
huggingface-cli download isabeth/rgad-crosslingual-tts-10h \\
  --repo-type dataset \\
  --local-dir data/rgad-crosslingual-tts-10h

python scripts/prepare_prefix_manifest.py \\
  --input-jsonl data/rgad-crosslingual-tts-10h/train.jsonl \\
  --output-dir data/rgad-crosslingual-tts-10h/prefix_manifest \\
  --split train \\
  --prompt-text-policy duration_filler \\
  --prompt-crop-seconds 6

python scripts/prepare_prefix_manifest.py \\
  --input-jsonl data/rgad-crosslingual-tts-10h/dev.jsonl \\
  --output-dir data/rgad-crosslingual-tts-10h/prefix_manifest \\
  --split dev \\
  --prompt-text-policy duration_filler \\
  --prompt-crop-seconds 6
```
"""


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-name", default="isabeth/rgad-crosslingual-tts-10h")
    parser.add_argument("--target-hours", type=float, default=10.0)
    parser.add_argument("--target-language", default="zh-CN")
    parser.add_argument("--dev-minutes", type=float, default=12.0)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = get_parser().parse_args()
    summary = build_dataset(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
