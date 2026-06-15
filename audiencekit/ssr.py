"""Semantic-similarity rating for mapping text responses to Likert PMFs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class TextEmbeddings(Protocol):
    """Minimal text embedding protocol used by SSR."""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


EmbeddingProvider = TextEmbeddings | Callable[[Sequence[str]], np.ndarray]


def _format_score(score: int | float) -> str:
    if isinstance(score, float) and score.is_integer():
        score = int(score)
    return str(score).replace("-", "minus_").replace(".", "_")


@dataclass(frozen=True)
class SSRAnchorSet:
    """Reference statements for one ordered rating scale."""

    id: str
    anchors: Mapping[int | float, str]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("SSRAnchorSet needs a non-empty id")
        if len(self.anchors) < 2:
            raise ValueError("SSRAnchorSet needs at least two anchors")
        normalized = {score: str(text).strip() for score, text in sorted(self.anchors.items())}
        empty_scores = [score for score, text in normalized.items() if not text]
        if empty_scores:
            raise ValueError(f"SSRAnchorSet {self.id!r} has empty anchors for {empty_scores}")
        object.__setattr__(self, "anchors", normalized)

    @property
    def scores(self) -> tuple[int | float, ...]:
        return tuple(self.anchors.keys())

    @property
    def statements(self) -> tuple[str, ...]:
        return tuple(self.anchors.values())


@dataclass(frozen=True)
class SSRResult:
    """Per-response SSR probability mass functions."""

    texts: tuple[str, ...]
    scores: tuple[int | float, ...]
    pmfs: np.ndarray
    reference_set_id: str

    @property
    def expected_scores(self) -> np.ndarray:
        return self.pmfs @ np.asarray(self.scores, dtype=float)

    @property
    def most_likely_scores(self) -> np.ndarray:
        if len(self.pmfs) == 0:
            return np.asarray([])
        score_values = np.asarray(self.scores)
        return score_values[np.argmax(self.pmfs, axis=1)]

    def survey_pmf(self, weights: Sequence[float] | None = None) -> np.ndarray:
        if len(self.pmfs) == 0:
            return np.zeros(len(self.scores), dtype=float)
        if weights is None:
            return self.pmfs.mean(axis=0)
        weight_array = np.asarray(weights, dtype=float)
        if weight_array.shape != (len(self.pmfs),):
            raise ValueError("weights must have one value per PMF row")
        if weight_array.sum() <= 0:
            raise ValueError("weights must sum to a positive value")
        return np.average(self.pmfs, axis=0, weights=weight_array)

    def to_frame(self, prefix: str = "ssr") -> pd.DataFrame:
        data: dict[str, object] = {
            f"{prefix}_text": list(self.texts),
            prefix: self.expected_scores,
            f"{prefix}_most_likely": self.most_likely_scores,
        }
        for idx, score in enumerate(self.scores):
            data[f"{prefix}_pmf_{_format_score(score)}"] = self.pmfs[:, idx]
        return pd.DataFrame(data)


class SentenceTransformerEmbeddings:
    """Lazy wrapper around sentence-transformers for local SSR embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ValueError(
                "Install AudienceKit with the 'ssr' extra to use default SSR embeddings: "
                "pip install 'audiencekit[ssr]'"
            ) from exc

        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts)), dtype=float)


class SemanticSimilarityRater:
    """Map free-text answers to probability distributions over an ordered scale.

    The rater is dataset-agnostic: callers provide anchor statements and an
    embedding provider. It implements the SSR flow from the cited paper at the
    abstraction level AudienceKit needs: embed response text, compare to anchor
    text, shift similarities by the row minimum, normalize to a PMF, optionally
    temperature-scale, and average across reference sets when requested.
    """

    def __init__(
        self,
        reference_sets: Sequence[SSRAnchorSet],
        *,
        embeddings: EmbeddingProvider | None = None,
        epsilon: float = 0.0,
        temperature: float = 1.0,
    ):
        if not reference_sets:
            raise ValueError("SemanticSimilarityRater needs at least one SSRAnchorSet")
        self.reference_sets = {anchor_set.id: anchor_set for anchor_set in reference_sets}
        if len(self.reference_sets) != len(reference_sets):
            raise ValueError("SSR anchor set ids must be unique")

        first_scale = reference_sets[0].scores
        if any(anchor_set.scores != first_scale for anchor_set in reference_sets):
            raise ValueError("All SSR anchor sets must use the same score scale")
        self.scores = first_scale

        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        self.epsilon = epsilon
        self.temperature = temperature

        self.embeddings = embeddings or SentenceTransformerEmbeddings()
        self.reference_embeddings = {
            anchor_set.id: self._encode(anchor_set.statements) for anchor_set in reference_sets
        }

    @classmethod
    def for_purchase_intent(
        cls,
        *,
        embeddings: EmbeddingProvider | None = None,
        epsilon: float = 0.0,
        temperature: float = 1.0,
    ) -> "SemanticSimilarityRater":
        return cls(
            purchase_intent_anchor_sets(),
            embeddings=embeddings,
            epsilon=epsilon,
            temperature=temperature,
        )

    @property
    def available_reference_sets(self) -> list[str]:
        return list(self.reference_sets)

    def get_reference_sentences(self, reference_set_id: str) -> tuple[str, ...]:
        return self.reference_sets[reference_set_id].statements

    def score_texts(
        self,
        texts: Sequence[str],
        *,
        reference_set_id: str = "mean",
        epsilon: float | None = None,
        temperature: float | None = None,
    ) -> SSRResult:
        text_tuple = tuple(str(text) for text in texts)
        if not text_tuple:
            return SSRResult(text_tuple, self.scores, np.empty((0, len(self.scores))), reference_set_id)
        response_embeddings = self._encode(text_tuple)
        pmfs = self.score_embeddings(
            response_embeddings,
            reference_set_id=reference_set_id,
            epsilon=epsilon,
            temperature=temperature,
        )
        return SSRResult(text_tuple, self.scores, pmfs, reference_set_id)

    def score_embeddings(
        self,
        response_embeddings: np.ndarray,
        *,
        reference_set_id: str = "mean",
        epsilon: float | None = None,
        temperature: float | None = None,
    ) -> np.ndarray:
        responses = np.asarray(response_embeddings, dtype=float)
        if responses.ndim == 1:
            responses = responses.reshape(1, -1)
        if responses.ndim != 2:
            raise ValueError("response_embeddings must be a 2D array")
        if len(responses) == 0:
            return np.empty((0, len(self.scores)))

        if reference_set_id.lower() == "mean":
            pmfs = np.asarray(
                [
                    self._pmfs_from_embeddings(responses, reference_embeddings, epsilon=epsilon)
                    for reference_embeddings in self.reference_embeddings.values()
                ]
            ).mean(axis=0)
        else:
            if reference_set_id not in self.reference_embeddings:
                raise KeyError(reference_set_id)
            pmfs = self._pmfs_from_embeddings(
                responses,
                self.reference_embeddings[reference_set_id],
                epsilon=epsilon,
            )

        return self._temperature_scale(pmfs, self.temperature if temperature is None else temperature)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if hasattr(self.embeddings, "encode"):
            raw = self.embeddings.encode(list(texts))
        elif callable(self.embeddings):
            raw = self.embeddings(list(texts))
        else:
            raise TypeError("embeddings must provide encode(texts) or be callable")

        array = np.asarray(raw, dtype=float)
        if array.ndim == 1 and len(texts) == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[0] != len(texts):
            raise ValueError("embedding provider must return a 2D array with one row per text")
        if not np.all(np.isfinite(array)):
            raise ValueError("embeddings must be finite")
        if np.any(np.linalg.norm(array, axis=1) == 0):
            raise ValueError("embeddings must not contain zero vectors")
        return array

    def _pmfs_from_embeddings(
        self,
        response_embeddings: np.ndarray,
        reference_embeddings: np.ndarray,
        *,
        epsilon: float | None,
    ) -> np.ndarray:
        if response_embeddings.shape[1] != reference_embeddings.shape[1]:
            raise ValueError("response and reference embeddings must have the same dimension")
        eps = self.epsilon if epsilon is None else epsilon
        if eps < 0:
            raise ValueError("epsilon must be non-negative")

        response_norms = np.linalg.norm(response_embeddings, axis=1, keepdims=True)
        reference_norms = np.linalg.norm(reference_embeddings, axis=1, keepdims=True)
        similarities = (response_embeddings / response_norms) @ (reference_embeddings / reference_norms).T
        similarities = (1.0 + similarities) / 2.0

        shifted = similarities - similarities.min(axis=1, keepdims=True)
        numerator = shifted + eps
        denominator = numerator.sum(axis=1, keepdims=True)
        uniform = np.full_like(numerator, 1.0 / numerator.shape[1])
        return np.divide(numerator, denominator, out=uniform, where=denominator > 0)

    def _temperature_scale(self, pmfs: np.ndarray, temperature: float) -> np.ndarray:
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if temperature == 1.0 or len(pmfs) == 0:
            return pmfs
        if temperature == 0.0:
            scaled = np.zeros_like(pmfs)
            scaled[np.arange(pmfs.shape[0]), np.argmax(pmfs, axis=1)] = 1.0
            uniform_rows = np.all(np.isclose(pmfs, pmfs[:, :1]), axis=1)
            scaled[uniform_rows] = pmfs[uniform_rows]
            return scaled
        scaled = pmfs ** (1.0 / temperature)
        return scaled / scaled.sum(axis=1, keepdims=True)


def purchase_intent_anchor_sets() -> tuple[SSRAnchorSet, ...]:
    """Default domain-neutral purchase-intent anchors for 1-5 SSR scoring."""

    return (
        SSRAnchorSet(
            "purchase_intent_plain",
            {
                1: "I definitely would not buy it.",
                2: "I probably would not buy it.",
                3: "I am unsure whether I would buy it.",
                4: "I probably would buy it.",
                5: "I definitely would buy it.",
            },
        ),
        SSRAnchorSet(
            "purchase_intent_likelihood",
            {
                1: "There is almost no chance I would purchase this.",
                2: "I would be unlikely to purchase this.",
                3: "I might or might not purchase this.",
                4: "I would be likely to purchase this.",
                5: "There is a very strong chance I would purchase this.",
            },
        ),
        SSRAnchorSet(
            "purchase_intent_consideration",
            {
                1: "This is not something I would consider buying.",
                2: "I would lean against buying this.",
                3: "I would need more information before deciding.",
                4: "I would lean toward buying this.",
                5: "This is something I would strongly consider buying.",
            },
        ),
    )
