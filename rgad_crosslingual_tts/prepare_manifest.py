"""Build prefix fine-tuning manifests from simple JSONL rows."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Any

from lhotse import CutSet
from lhotse.audio import Recording
from lhotse.cut import MonoCut
from lhotse.supervision import SupervisionSegment

from rgad_crosslingual_tts.audio import build_duration_filler, concat_prompt_target
from rgad_crosslingual_tts.constants import DEFAULT_PROMPT_SECONDS, DEFAULT_SAMPLE_RATE


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_prefix_manifest(
    input_jsonl: str | Path,
    output_dir: str | Path,
    *,
    split: str,
    prompt_crop_seconds: float | None = DEFAULT_PROMPT_SECONDS,
    prompt_trailing_silence_seconds: float = 0.2,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    prompt_text_policy: str = "duration_filler",
) -> dict[str, Any]:
    """Build a Lhotse CutSet manifest for cross-lingual prefix fine-tuning."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    materialized_rows: list[dict[str, Any]] = []
    cuts = []

    for index, row in enumerate(read_jsonl(input_jsonl)):
        sample_id = str(row.get("id") or row.get("sample_id") or f"{split}_{index:06d}")
        prompt_wav = Path(str(row["prompt_wav"])).expanduser().resolve()
        target_wav = Path(str(row["target_wav"])).expanduser().resolve()
        target_text = " ".join(str(row["text"]).split())
        prompt_language = str(row.get("prompt_language") or "und")
        target_language = str(row.get("target_language") or "zh-CN")
        speaker_id = str(row.get("speaker_id") or sample_id)
        concat_wav = output_dir / "concat_wavs" / split / f"{_slug(sample_id)}.wav"
        durations = concat_prompt_target(
            prompt_wav,
            target_wav,
            concat_wav,
            prompt_crop_seconds=prompt_crop_seconds,
            prompt_trailing_silence_seconds=prompt_trailing_silence_seconds,
            sample_rate=sample_rate,
        )
        if prompt_text_policy == "duration_filler":
            prompt_text = build_duration_filler(durations["prompt_audio_duration_seconds"])
        elif prompt_text_policy == "original":
            prompt_text = " ".join(str(row.get("prompt_text") or "").split())
            if not prompt_text:
                raise ValueError(f"{sample_id}: prompt_text is required with policy=original")
        else:
            raise ValueError(f"unsupported prompt_text_policy: {prompt_text_policy}")

        recording = Recording.from_file(str(concat_wav), recording_id=_slug(sample_id))
        custom = {
            "prompt_audio": str(prompt_wav),
            "prompt_text": prompt_text,
            "prompt_language": prompt_language,
            "prompt_audio_duration_seconds": durations["prompt_audio_duration_seconds"],
            "prompt_condition_duration_seconds": durations["prompt_condition_duration_seconds"],
            "target_audio": str(target_wav),
            "target_text": target_text,
            "target_language": target_language,
            "target_duration_seconds": durations["target_duration_seconds"],
            "concat_audio_path": str(concat_wav),
        }
        supervision = SupervisionSegment(
            id=sample_id,
            recording_id=recording.id,
            start=0.0,
            duration=recording.duration,
            channel=0,
            text=f"{prompt_text} {target_text}",
            language=target_language,
            speaker=speaker_id,
            custom=custom,
        )
        cut = MonoCut(
            id=sample_id,
            start=0.0,
            duration=recording.duration,
            channel=0,
            supervisions=[supervision],
            recording=recording,
            custom=custom,
        )
        cuts.append(cut)
        materialized_rows.append(
            {
                "id": sample_id,
                "prompt_wav": str(prompt_wav),
                "target_wav": str(target_wav),
                "concat_wav": str(concat_wav),
                "prompt_text": prompt_text,
                "text": target_text,
                "prompt_language": prompt_language,
                "target_language": target_language,
                "speaker_id": speaker_id,
                **durations,
            }
        )

    rows_path = output_dir / f"{split}.jsonl"
    cuts_path = output_dir / f"cuts_{split}.jsonl.gz"
    write_jsonl(rows_path, materialized_rows)
    CutSet.from_cuts(cuts).to_file(cuts_path)
    summary = {
        "split": split,
        "num_rows": len(materialized_rows),
        "rows": str(rows_path),
        "cuts": str(cuts_path),
        "prompt_text_policy": prompt_text_policy,
        "prompt_crop_seconds": prompt_crop_seconds,
    }
    (output_dir / f"{split}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "item"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--prompt-crop-seconds", type=float, default=DEFAULT_PROMPT_SECONDS)
    parser.add_argument("--prompt-trailing-silence-seconds", type=float, default=0.2)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument(
        "--prompt-text-policy",
        choices=["duration_filler", "original"],
        default="duration_filler",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = get_parser().parse_args(argv)
    summary = build_prefix_manifest(
        args.input_jsonl,
        args.output_dir,
        split=args.split,
        prompt_crop_seconds=args.prompt_crop_seconds,
        prompt_trailing_silence_seconds=args.prompt_trailing_silence_seconds,
        sample_rate=args.sample_rate,
        prompt_text_policy=args.prompt_text_policy,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
