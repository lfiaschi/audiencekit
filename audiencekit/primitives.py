"""Dataset-agnostic primitives for synthetic audience studies."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class _UnknownDict(defaultdict):
    def __missing__(self, key):
        return self.default_factory()


@dataclass(frozen=True)
class PersonaTemplate:
    """Render a persona prompt from any row-like mapping."""

    template: str
    missing_value: str = "Unknown"

    def render(self, attributes: Mapping[str, Any]) -> str:
        values = _UnknownDict(lambda: self.missing_value)
        values.update({key: self._clean(value) for key, value in attributes.items()})
        return self.template.format_map(values)

    def _clean(self, value: Any) -> str:
        if value is None:
            return self.missing_value
        try:
            if pd.isna(value):
                return self.missing_value
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        return text or self.missing_value


@dataclass(frozen=True)
class AudienceFrame:
    """A weighted respondent frame independent of any specific data source."""

    data: pd.DataFrame
    id_column: str = "id"
    weight_column: str = "weight"

    def __post_init__(self) -> None:
        if self.id_column not in self.data.columns:
            raise ValueError(f"AudienceFrame needs id column {self.id_column!r}")

    def sample(
        self,
        n: int,
        *,
        segment: Callable[[Mapping[str, Any]], bool] | None = None,
        segment_name: str = "broad",
        seed: int = 42,
        weighted: bool = True,
        replace: bool = False,
    ) -> pd.DataFrame:
        """Draw a respondent sample, optionally filtered by a row predicate.

        Raises ``ValueError`` when ``n`` exceeds the number of matching
        respondents (or, if ``weighted``, the number with positive weight)
        unless ``replace=True`` is passed to explicitly allow sampling with
        replacement.
        """
        if n <= 0:
            raise ValueError("n must be positive")

        pool = self.data.copy()
        if segment is not None:
            pool = pool[pool.apply(segment, axis=1)]
        if pool.empty:
            raise ValueError(f"No respondents available for segment {segment_name!r}")

        available = len(pool)
        if weighted:
            if self.weight_column not in pool.columns:
                raise ValueError(f"weighted=True needs a {self.weight_column!r} column")
            weights = pd.to_numeric(pool[self.weight_column], errors="coerce").fillna(0.0).to_numpy()
            if weights.sum() <= 0:
                raise ValueError(f"weighted=True needs positive values in {self.weight_column!r}")
            available = int((weights > 0).sum())

        if n > available and not replace:
            raise ValueError(
                f"n={n} exceeds the {available} matching respondents; "
                "pass replace=True to sample with replacement"
            )

        if weighted:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(pool), size=n, replace=replace or n > available, p=weights / weights.sum())
            sampled = pool.iloc[idx]
        else:
            sampled = pool.sample(n=n, random_state=seed, replace=replace or n > available)

        sampled = sampled.reset_index(drop=True)
        sampled["segment"] = segment_name
        return sampled
