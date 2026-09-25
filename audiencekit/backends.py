"""LLM backends for synthetic audience studies: Gemini, OpenAI, and Anthropic.

Both support an optional local image attached to the prompt (vision input),
which is how stimulus images reach the synthetic respondent.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

MAX_RETRIES = 3
BASE_DELAY = 1.0


@dataclass(frozen=True)
class Media:
    """One stimulus attachment: raw bytes, or a provider file reference URI.

    ``data`` is either raw bytes for an inline attachment, or a ``str`` URI
    for a provider file reference (e.g. the Gemini Files API ``uri``, or a
    YouTube URL).
    """

    data: Union[bytes, str]
    mime_type: str

    @property
    def is_video(self) -> bool:
        return self.mime_type.startswith("video/")


def encode_image(image_path: Union[str, Path]) -> tuple[str, str]:
    """Return (base64 payload, mime type) for a local image file."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    mime_type = mimetypes.guess_type(path)[0]
    if not mime_type:
        raise ValueError(f"Cannot determine mime type for: {path}")
    return base64.b64encode(path.read_bytes()).decode("utf-8"), mime_type


def media_from_path(path: Union[str, Path]) -> Media:
    """Build a :class:`Media` from a local file's bytes."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {p}")
    mime_type = mimetypes.guess_type(p)[0]
    if not mime_type:
        raise ValueError(f"Cannot determine mime type for: {p}")
    return Media(p.read_bytes(), mime_type)


def _as_media(
    image: Optional[Union[str, Path]], media: Sequence[Media]
) -> tuple[Media, ...]:
    """Merge the deprecated ``image=`` alias into the ``media`` sequence."""
    items = tuple(media or ())
    if image:
        items = (media_from_path(image),) + items
    return items


class LLMBackend(ABC):
    """Minimal completion interface shared by all providers."""

    env_var: str = ""
    env_vars: tuple[str, ...] = ()

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        env_vars = self.env_vars or (self.env_var,)
        self.api_key = api_key or next((os.getenv(name) for name in env_vars if os.getenv(name)), None)
        if not self.api_key:
            joined = " or ".join(env_vars)
            raise ValueError(f"Set {joined} or pass api_key")
        self.model = model
        self._initialize_client()

    @abstractmethod
    def _initialize_client(self) -> None: ...

    def _check_media(self, media: Sequence[Media]) -> None:
        """Raise ValueError for attachments this provider cannot accept."""

    @abstractmethod
    def _complete(
        self,
        prompt: str,
        image: Optional[Union[str, Path]],
        *,
        media: Sequence[Media] = (),
        **kwargs: Any,
    ) -> str: ...

    def get_completion(
        self,
        prompt: str,
        image: Optional[Union[str, Path]] = None,
        *,
        media: Sequence[Media] = (),
        **kwargs: Any,
    ) -> str:
        """Completion with exponential-backoff retry.

        ``image=`` is a deprecated alias for a single local-file attachment;
        it is converted into one :class:`Media` item ahead of ``media``.
        """
        items = _as_media(image, media)
        # Unsupported media is a deterministic caller error: fail fast, don't retry.
        self._check_media(items)
        for attempt in range(MAX_RETRIES + 1):
            try:
                return self._complete(prompt, None, media=items, **kwargs)
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(f"{type(self).__name__} failed after {MAX_RETRIES} retries: {exc}")
                time.sleep(BASE_DELAY * 2**attempt)
        raise RuntimeError("unreachable")


class OpenAIBackend(LLMBackend):
    env_var = "OPENAI_API_KEY"

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        super().__init__(api_key=api_key, model=model)

    def _initialize_client(self) -> None:
        import openai

        self.client = openai.OpenAI(api_key=self.api_key)

    def _check_media(self, media: Sequence[Media]) -> None:
        if any(item.is_video or not isinstance(item.data, bytes) for item in media):
            raise ValueError("OpenAIBackend does not support video or file-reference media")

    def _complete(
        self,
        prompt: str,
        image: Optional[Union[str, Path]],
        *,
        media: Sequence[Media] = (),
        **kwargs: Any,
    ) -> str:
        content: Any = [{"type": "text", "text": prompt}]
        self._check_media(media)
        for item in media:
            payload = base64.b64encode(item.data).decode("utf-8")
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:{item.mime_type};base64,{payload}"}}
            )
        if not media:
            content = prompt
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=kwargs.pop("max_tokens", 1024),
            temperature=kwargs.pop("temperature", 0.7),
            **kwargs,
        )
        return response.choices[0].message.content


class GeminiBackend(LLMBackend):
    env_var = "GEMINI_API_KEY"
    env_vars = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        super().__init__(api_key=api_key, model=model)

    def _initialize_client(self) -> None:
        from google import genai

        self.client = genai.Client(api_key=self.api_key)

    def _complete(
        self,
        prompt: str,
        image: Optional[Union[str, Path]],
        *,
        media: Sequence[Media] = (),
        **kwargs: Any,
    ) -> str:
        from google.genai import types

        parts: list[Any] = [prompt]
        for item in media:
            if isinstance(item.data, bytes):
                parts.append(types.Part.from_bytes(data=item.data, mime_type=item.mime_type))
            else:
                parts.append(
                    types.Part(file_data=types.FileData(file_uri=item.data, mime_type=item.mime_type))
                )
        contents: Any = parts if media else prompt

        config = types.GenerateContentConfig(
            max_output_tokens=kwargs.pop("max_tokens", 1024),
            temperature=kwargs.pop("temperature", 0.7),
            **kwargs,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return response.text or ""


class AnthropicBackend(LLMBackend):
    env_var = "ANTHROPIC_API_KEY"

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-haiku-4-5-20251001"):
        super().__init__(api_key=api_key, model=model)

    def _initialize_client(self) -> None:
        import anthropic

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def _check_media(self, media: Sequence[Media]) -> None:
        if any(item.is_video or not isinstance(item.data, bytes) for item in media):
            raise ValueError("AnthropicBackend does not support video or file-reference media")

    def _complete(
        self,
        prompt: str,
        image: Optional[Union[str, Path]],
        *,
        media: Sequence[Media] = (),
        **kwargs: Any,
    ) -> str:
        content: Any = []
        self._check_media(media)
        for item in media:
            payload = base64.b64encode(item.data).decode("utf-8")
            content.append(
                {"type": "image", "source": {"type": "base64", "media_type": item.mime_type, "data": payload}}
            )
        content.append({"type": "text", "text": prompt})
        message = self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=kwargs.pop("max_tokens", 1024),
            temperature=kwargs.pop("temperature", 0.7),
            **kwargs,
        )
        return "".join(block.text for block in message.content if block.type == "text")


def make_backend(backend_type: str = "gemini", model: Optional[str] = None) -> LLMBackend:
    if backend_type == "gemini":
        return GeminiBackend(model=model or "gemini-2.5-flash")
    if backend_type == "openai":
        return OpenAIBackend(model=model or "gpt-4o-mini")
    if backend_type == "anthropic":
        return AnthropicBackend(model=model or "claude-haiku-4-5-20251001")
    raise ValueError(f"Unsupported backend: {backend_type!r} (use 'gemini', 'openai', or 'anthropic')")
