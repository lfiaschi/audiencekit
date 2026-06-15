# Data And Asset Notice

AudienceKit
Copyright 2026 Luca Fiaschi

The Apache License 2.0 covers the AudienceKit source code.

## Bundled GSS Sample Panel

`audiencekit/data/gss_panel_2022.csv` is a prepared educational sample derived
from public General Social Survey data. It is included so examples and smoke
tests work without downloading the full cumulative file.

For production work, download the current GSS cumulative file from NORC and
prepare your own panel with `audiencekit.gss.load_gss` or
`scripts/extract_panel.py`.

Recommended GSS citation:

> Davern, Michael; Bautista, Rene; Freese, Jeremy; Herd, Pamela; and Morgan,
> Stephen L. General Social Survey 1972-2024. NORC, Chicago, 2026.

Review NORC's data access terms before redistributing derived data:
https://gss.norc.org/get-the-data.html

## Ferrari Luce Example

The Ferrari Luce example is an educational synthetic research case study.
Ferrari and Prancing Horse marks belong to their owners. Example stimuli are
not part of the AudienceKit software license. Replace them with your own
properly licensed assets before using the example for public, commercial, or
redistributed work.
