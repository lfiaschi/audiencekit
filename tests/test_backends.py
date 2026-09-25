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


class FakeFileData:
    def __init__(self, file_uri, mime_type=None):
        self.file_uri = file_uri
        self.mime_type = mime_type


class FakePart:
    def __init__(self, file_data=None):
        self.file_data = file_data

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
    types_module.FileData = FakeFileData
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


def test_gemini_media_list_maps_bytes_and_uris(monkeypatch) -> None:
    from audiencekit import Media

    install_fake_genai(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    backend = make_backend()

    backend.get_completion(
        "hi",
        media=[Media(b"jpegbytes", "image/jpeg"), Media("https://youtu.be/x", "video/mp4")],
    )

    contents = FakeClient.last_client.models.calls[0]["contents"]
    assert contents[0] == "hi"
    assert contents[1] == {"data": b"jpegbytes", "mime_type": "image/jpeg"}
    assert contents[2].file_data.file_uri == "https://youtu.be/x"


def test_image_alias_becomes_media(monkeypatch, tmp_path) -> None:
    install_fake_genai(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    img = tmp_path / "a.png"
    img.write_bytes(b"png")
    backend = make_backend()

    backend.get_completion("hi", image=img)

    contents = FakeClient.last_client.models.calls[0]["contents"]
    assert contents[1] == {"data": b"png", "mime_type": "image/png"}


def test_openai_rejects_video(monkeypatch) -> None:
    import pytest
    from audiencekit import Media
    from audiencekit.backends import OpenAIBackend

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setattr(OpenAIBackend, "_initialize_client", lambda self: None)
    backend = OpenAIBackend()

    with pytest.raises(ValueError, match="video"):
        backend._complete("hi", None, media=(Media(b"x", "video/mp4"),))


class FakeAnthropicMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="anthropic says hi")]
        )


class FakeAnthropicClient:
    def __init__(self):
        self.messages = FakeAnthropicMessages()


def test_anthropic_rejects_video(monkeypatch) -> None:
    import pytest
    from audiencekit import Media
    from audiencekit.backends import AnthropicBackend

    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    monkeypatch.setattr(AnthropicBackend, "_initialize_client", lambda self: None)
    backend = AnthropicBackend()

    with pytest.raises(ValueError, match="video"):
        backend._complete("hi", None, media=(Media(b"x", "video/mp4"),))


def test_anthropic_maps_image_media(monkeypatch) -> None:
    from audiencekit import Media
    from audiencekit.backends import AnthropicBackend

    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    monkeypatch.setattr(AnthropicBackend, "_initialize_client", lambda self: None)
    backend = AnthropicBackend()
    backend.client = FakeAnthropicClient()

    result = backend.get_completion("hi", media=[Media(b"png", "image/png")])

    assert result == "anthropic says hi"
    call = backend.client.messages.calls[0]
    content = call["messages"][0]["content"]
    assert content[0] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "cG5n"},
    }
    assert content[1] == {"type": "text", "text": "hi"}


def test_unsupported_media_fails_fast_without_retry(monkeypatch) -> None:
    import pytest
    from audiencekit import Media
    from audiencekit.backends import AnthropicBackend, OpenAIBackend

    def no_sleep(_seconds):
        raise AssertionError("unsupported media must not be retried")

    monkeypatch.setattr("audiencekit.backends.time.sleep", no_sleep)
    for cls, env in ((OpenAIBackend, "OPENAI_API_KEY"), (AnthropicBackend, "ANTHROPIC_API_KEY")):
        monkeypatch.setenv(env, "key")
        monkeypatch.setattr(cls, "_initialize_client", lambda self: None)
        backend = cls()
        with pytest.raises(ValueError, match="video"):
            backend.get_completion("hi", media=[Media("https://example.com/v.mp4", "video/mp4")])


class FakeOpenAICompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = types.SimpleNamespace(content="openai says hi")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def test_openai_maps_image_media(monkeypatch) -> None:
    from audiencekit import Media
    from audiencekit.backends import OpenAIBackend

    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setattr(OpenAIBackend, "_initialize_client", lambda self: None)
    backend = OpenAIBackend()
    completions = FakeOpenAICompletions()
    backend.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))

    result = backend.get_completion("hi", media=[Media(b"png", "image/png")])

    assert result == "openai says hi"
    content = completions.calls[0]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "hi"}
    assert content[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,cG5n"}}
