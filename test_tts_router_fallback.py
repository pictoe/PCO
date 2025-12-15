import threading


def test_tts_router_falls_back_from_piper_to_sapi(monkeypatch) -> None:
    # Patch AudioPlayer to avoid using sounddevice.
    class _FakePlayer:
        def __init__(self, device=None):
            self.device = device

        def play_wav(self, wav_path: str, stop_event: threading.Event) -> bool:
            return True

    # Fake Piper engine: first model fails, second model works.
    class _FakePiper:
        def __init__(self, exe_path: str, model_path: str, speaker_id=None, length_scale=1.0, sentence_silence=0.25):
            self.model_path = model_path

        def synthesize_to_wav(self, text: str) -> str:
            # Simulate a model crash / empty output for non-lessac models.
            if "lessac" not in (self.model_path or ""):
                return ""
            return __file__  # any existing path to satisfy cleanup attempt is patched out below

    # Fake SAPI engine: always works.
    class _FakeSapi:
        def __init__(self, voice_name: str = "", rate: float = 1.0):
            self.voice_name = voice_name

        def synthesize_to_wav(self, text: str) -> str:
            return __file__

    import wyzer.tts.tts_router as tts_router

    monkeypatch.setattr(tts_router, "AudioPlayer", _FakePlayer)
    monkeypatch.setattr(tts_router, "PiperTTSEngine", _FakePiper)

    # Patch lazy import inside _init_engine by injecting into module namespace.
    import wyzer.tts.sapi_engine as sapi_engine

    monkeypatch.setattr(sapi_engine, "SapiTTSEngine", _FakeSapi)

    # Avoid deleting __file__ during cleanup.
    monkeypatch.setattr(tts_router.os, "unlink", lambda _p: None)

    r = tts_router.TTSRouter(
        engine="piper",
        piper_exe_path="piper.exe",
        piper_model_path="./wyzer/assets/piper/en_US-hfc_male-medium.onnx",
        sapi_voice_name="male",
        enabled=True,
    )

    stop_event = threading.Event()
    ok = r.speak("hello", stop_event)
    assert ok is True
    # Should have switched to Lessac piper after failure, or to SAPI; either is acceptable.
    assert r.engine_name in {"piper", "sapi"}
