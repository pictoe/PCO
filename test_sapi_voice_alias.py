from wyzer.tts.sapi_engine import pick_sapi_voice_by_gender


def test_pick_sapi_voice_by_gender_male() -> None:
    voices = [
        {"name": "Voice A", "gender": "Female", "culture": "en-US", "age": "Adult"},
        {"name": "Voice B", "gender": "Male", "culture": "en-US", "age": "Adult"},
    ]
    assert pick_sapi_voice_by_gender(voices, "male") == "Voice B"


def test_pick_sapi_voice_by_gender_prefers_neural_if_present() -> None:
    voices = [
        {"name": "Microsoft David Desktop", "gender": "Male", "culture": "en-US"},
        {"name": "Microsoft John Neural", "gender": "Male", "culture": "en-US"},
    ]
    assert pick_sapi_voice_by_gender(voices, "male") == "Microsoft John Neural"


def test_pick_sapi_voice_by_gender_female() -> None:
    voices = [
        {"name": "Voice A", "gender": "Female"},
        {"name": "Voice B", "gender": "Male"},
    ]
    assert pick_sapi_voice_by_gender(voices, "female") == "Voice A"


def test_pick_sapi_voice_by_gender_unknown_returns_empty() -> None:
    voices = [{"name": "Voice A", "gender": "Female"}]
    assert pick_sapi_voice_by_gender(voices, "robot") == ""
