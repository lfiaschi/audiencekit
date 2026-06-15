from __future__ import annotations

import numpy as np
import pytest

from audiencekit import SSRAnchorSet, SemanticSimilarityRater


class TinyEmbeddings:
    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def encode(self, texts):
        return np.array([self.vectors[text] for text in texts], dtype=float)


def assert_valid_pmfs(pmfs: np.ndarray) -> None:
    assert np.allclose(pmfs.sum(axis=1), 1.0)
    assert np.all(pmfs >= 0)


def test_ssr_scores_text_against_custom_anchor_set() -> None:
    anchors = SSRAnchorSet(
        "purchase",
        {
            1: "I definitely would not buy it.",
            2: "I probably would not buy it.",
            3: "I am unsure whether I would buy it.",
            4: "I probably would buy it.",
            5: "I definitely would buy it.",
        },
    )
    embeddings = TinyEmbeddings(
        {
            "I definitely would not buy it.": [1, 0, 0, 0, 0],
            "I probably would not buy it.": [0, 1, 0, 0, 0],
            "I am unsure whether I would buy it.": [0, 0, 1, 0, 0],
            "I probably would buy it.": [0, 0, 0, 1, 0],
            "I definitely would buy it.": [0, 0, 0, 0, 1],
            "I would absolutely buy this.": [0, 0, 0, 0, 1],
        }
    )

    rater = SemanticSimilarityRater([anchors], embeddings=embeddings)
    result = rater.score_texts(["I would absolutely buy this."], reference_set_id="purchase")

    assert_valid_pmfs(result.pmfs)
    assert result.scores == (1, 2, 3, 4, 5)
    assert result.expected_scores.tolist() == [5.0]
    assert result.most_likely_scores.tolist() == [5]
    assert result.to_frame(prefix="fit").loc[0, "fit_pmf_5"] == pytest.approx(1.0)


def test_ssr_mean_reference_set_averages_across_anchor_sets() -> None:
    set_a = SSRAnchorSet("a", {1: "a1", 2: "a2", 3: "a3", 4: "a4", 5: "a5"})
    set_b = SSRAnchorSet("b", {1: "b1", 2: "b2", 3: "b3", 4: "b4", 5: "b5"})
    vectors = {
        "a1": [1, 0, 0, 0, 0],
        "a2": [0, 1, 0, 0, 0],
        "a3": [0, 0, 1, 0, 0],
        "a4": [0, 0, 0, 1, 0],
        "a5": [0, 0, 0, 0, 1],
        "b1": [1, 0, 0, 0, 0],
        "b2": [0, 1, 0, 0, 0],
        "b3": [0, 0, 1, 0, 0],
        "b4": [0, 0, 0, 0, 1],
        "b5": [0, 0, 0, 1, 0],
        "clear yes": [0, 0, 0, 0, 1],
    }

    rater = SemanticSimilarityRater([set_a, set_b], embeddings=TinyEmbeddings(vectors))
    result = rater.score_texts(["clear yes"], reference_set_id="mean")

    assert_valid_pmfs(result.pmfs)
    assert result.expected_scores.tolist() == [4.5]
    assert result.pmfs.tolist() == [[0.0, 0.0, 0.0, 0.5, 0.5]]


def test_ssr_rejects_anchor_sets_with_inconsistent_scales() -> None:
    with pytest.raises(ValueError, match="same score scale"):
        SemanticSimilarityRater(
            [
                SSRAnchorSet("five", {1: "no", 2: "maybe no", 3: "maybe", 4: "maybe yes", 5: "yes"}),
                SSRAnchorSet("three", {1: "no", 2: "maybe", 3: "yes"}),
            ],
            embeddings=TinyEmbeddings({}),
        )


def test_ssr_rejects_invalid_precomputed_response_embeddings() -> None:
    anchors = SSRAnchorSet("purchase", {1: "no", 2: "maybe no", 3: "maybe", 4: "maybe yes", 5: "yes"})
    rater = SemanticSimilarityRater(
        [anchors],
        embeddings=TinyEmbeddings(
            {
                "no": [1, 0, 0, 0, 0],
                "maybe no": [0, 1, 0, 0, 0],
                "maybe": [0, 0, 1, 0, 0],
                "maybe yes": [0, 0, 0, 1, 0],
                "yes": [0, 0, 0, 0, 1],
            }
        ),
    )

    with pytest.raises(ValueError, match="finite"):
        rater.score_embeddings(np.array([[np.nan, 0, 0, 0, 1]]), reference_set_id="purchase")

    with pytest.raises(ValueError, match="zero vectors"):
        rater.score_embeddings(np.zeros((1, 5)), reference_set_id="purchase")


def test_survey_pmf_rejects_negative_and_nonfinite_weights() -> None:
    anchors = SSRAnchorSet("purchase", {1: "no", 2: "maybe no", 3: "maybe", 4: "maybe yes", 5: "yes"})
    rater = SemanticSimilarityRater(
        [anchors],
        embeddings=TinyEmbeddings(
            {
                "no": [1, 0, 0, 0, 0],
                "maybe no": [0, 1, 0, 0, 0],
                "maybe": [0, 0, 1, 0, 0],
                "maybe yes": [0, 0, 0, 1, 0],
                "yes": [0, 0, 0, 0, 1],
                "clear yes": [0, 0, 0, 0, 1],
                "clear no": [1, 0, 0, 0, 0],
            }
        ),
    )
    result = rater.score_texts(["clear yes", "clear no"], reference_set_id="purchase")

    with pytest.raises(ValueError, match="non-negative"):
        result.survey_pmf(weights=[1.0, -0.5])

    with pytest.raises(ValueError, match="finite"):
        result.survey_pmf(weights=[1.0, np.nan])
