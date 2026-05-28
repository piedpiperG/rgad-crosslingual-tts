from rgad_crosslingual_tts.audio import build_duration_filler
from rgad_crosslingual_tts.prepare_manifest import _resolve_audio_path


def test_build_duration_filler() -> None:
    assert build_duration_filler(1.0) == "嗯嗯嗯嗯。"
    assert build_duration_filler(0.0) == "嗯。"


def test_resolve_audio_path_relative_to_manifest(tmp_path) -> None:
    manifest_dir = tmp_path / "dataset"
    audio_path = manifest_dir / "audio" / "sample.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.touch()

    assert _resolve_audio_path("audio/sample.wav", manifest_dir) == audio_path.resolve()
