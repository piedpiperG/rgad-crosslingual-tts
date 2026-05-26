from rgad_crosslingual_tts.audio import build_duration_filler


def test_build_duration_filler() -> None:
    assert build_duration_filler(1.0) == "嗯嗯嗯嗯。"
    assert build_duration_filler(0.0) == "嗯。"
