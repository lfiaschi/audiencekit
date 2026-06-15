from __future__ import annotations

import sys
import types

from audiencekit.backends import GeminiBackend, make_backend


class FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(text="gemini says hi")


class FakeClient:
    last_client = None

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.models = FakeModels()
        FakeClient.last_client = self


class FakeGenerateContentConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakePart:
    @classmethod
    def from_bytes(cls, data, mime_type):
        return {"data": data, "mime_type": mime_type}


def install_fake_genai(monkeypatch):
    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    types_module = types.ModuleType("google.genai.types")

    genai_module.Client = FakeClient
    genai_module.types = types_module
    types_module.GenerateContentConfig = FakeGenerateContentConfig
    types_module.Part = FakePart
    google_module.genai = genai_module

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)


def test_make_backend_defaults_to_gemini_flash(monkeypatch) -> None:
    install_fake_genai(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "key")

    backend = make_backend()

    assert isinstance(backend, GeminiBackend)
    assert backend.model == "gemini-2.5-flash"
    assert FakeClient.last_client.api_key == "key"


def test_gemini_backend_accepts_google_api_key(monkeypatch) -> None:
    install_fake_genai(monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")

    backend = GeminiBackend()

    assert backend.api_key == "google-key"
    assert FakeClient.last_client.api_key == "google-key"


def test_gemini_backend_generate_content_uses_config(monkeypatch) -> None:
    install_fake_genai(monkeypatch)

    backend = GeminiBackend(api_key="key")
    result = backend.get_completion("hello", temperature=0.2, max_tokens=123)

    assert result == "gemini says hi"
    call = FakeClient.last_client.models.calls[0]
    assert call["model"] == "gemini-2.5-flash"
    assert call["contents"] == "hello"
    assert call["config"].kwargs == {"temperature": 0.2, "max_output_tokens": 123}


def test_gemini_backend_attaches_image(monkeypatch, tmp_path) -> None:
    install_fake_genai(monkeypatch)
    image = tmp_path / "stimulus.png"
    image.write_bytes(b"png")

    backend = GeminiBackend(api_key="key")
    backend.get_completion("describe", image=image)

    contents = FakeClient.last_client.models.calls[0]["contents"]
    assert contents[0] == "describe"
    assert contents[1] == {"data": b"png", "mime_type": "image/png"}
