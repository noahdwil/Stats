# Seattle / Las Vegas Expansion Scenarios — Methodology Notes

## Before anything else: what I found (and didn't find)

I searched the full project for the existing DMA-percentile-mapping method before
changing anything: `nba_master_model_data.csv`, `nba_historical_deals.csv`,
`NBA_Jersey_Patch_Valuation.xlsx` (including its cell formulas, not just the
cached values), every `.py` file in `jersey_patch_model/`, and the repository's
other git branch. **No file anywhere in this project contains the old
Seattle/Las Vegas calculation.** The description of the old method used below
comes only from how it was described in the request, not from inspecting
working code. No old numbers exist in the project to preserve or compare
against directly — there is nothing being silently changed, because there was
nothing numeric here to begin with.

## Comparison set used in Stage 1

**All current NBA teams except LA Lakers and Golden State Warriors (n = 28).**

Rationale: those two teams' `Exposure_Value_Zoomph` reflects decades of
global-brand history (international fanbases, championship pedigree, media
markets built over 50+ years) that no expansion team starts with in its first
season. Including them would pull the percentile benchmarks upward based on
two extreme outliers unrepresentative of a new franchise's plausible starting
point. Every other current team — across every performance and exposure
tier, large market and small — is included, so the comparison set isn't
narrowed by any further judgment call about which teams are "similar enough."

## Stage 1 — Exposure scenarios

```
comparison_set = all 30 current teams EXCEPT (LA Lakers, Golden State Warriors)   # n = 28
Low (P25)     = percentile(comparison_set.Exposure_Value_Zoomph, 25) = $13,625,000
Typical (P50) = percentile(comparison_set.Exposure_Value_Zoomph, 50) = $21,000,000
High (P75)    = percentile(comparison_set.Exposure_Value_Zoomph, 75) = $29,825,000
```

DMA rank and TV-homes counts are **not** inputs to this calculation. Because of
that, Seattle and Las Vegas receive the **same three exposure-scenario dollar
figures** — the distinction between the two cities is qualitative only (DMA
rank, TV homes, prior pro-sports history), shown as text context on the chart,
never used to shift a number.

## Stage 2 — Pricing conversion

The four disclosed real-deal ratios (price ÷ exposure), as given:

| Team | Ratio |
|---|---|
| Warriors | 0.26 |
| Lakers | 0.196 |
| Mavericks | 0.35 |
| 76ers | 0.64 |

```
sorted ratios = [0.196, 0.26, 0.35, 0.64]
median (n=4, even) = (0.26 + 0.35) / 2 = 0.305
```

- **Historical low** = exposure × 0.196
- **Observed midpoint** = exposure × 0.305
- **Historical high** = exposure × 0.64

These are the as-observed ratios from 4 disclosed deals — not a statistically
fitted coefficient, not a confidence interval, and not a prediction.

## Output 1 — Calculation table

See `expansion_scenario_table.csv`. Reproduced here:

| City | Exposure scenario | Exposure assumption | 0.196× valuation | Observed-median valuation | 0.64× valuation |
|---|---|---|---|---|---|
| Seattle | Low (P25) | $13,625,000 | $2,670,500 | $4,155,625 | $8,720,000 |
| Seattle | Typical (P50) | $21,000,000 | $4,116,000 | $6,405,000 | $13,440,000 |
| Seattle | High (P75) | $29,825,000 | $5,845,700 | $9,096,625 | $19,088,000 |
| Las Vegas | Low (P25) | $13,625,000 | $2,670,500 | $4,155,625 | $8,720,000 |
| Las Vegas | Typical (P50) | $21,000,000 | $4,116,000 | $6,405,000 | $13,440,000 |
| Las Vegas | High (P75) | $29,825,000 | $5,845,700 | $9,096,625 | $19,088,000 |

## Output 2 — Visualization

`expansion_scenario_chart.png` — two panels (Seattle, Las Vegas), each showing
the three exposure scenarios as horizontal range bars (historical-low to
historical-high), with a diamond marker at the observed-median value. DMA
rank/TV homes appear as an italic text line above each panel title, visually
separated from the chart math to reinforce that they're context, not inputs.

## Output 3 — Suggested poster language

> **Illustrative expansion-market scenarios, not forecasts.** The NBA has not
> finalized expansion to Seattle or Las Vegas. These figures describe
> *conditional* "what if a team existed there" scenarios built from current
> league data — they are not predictions of what either market's patch value
> would actually be.
>
> **Exposure scenarios** (Low / Typical / High) are the 25th/50th/75th
> percentile of current teams' media-exposure values (excluding the Lakers and
> Warriors, whose exposure reflects decades of brand history an expansion team
> wouldn't start with). Seattle and Las Vegas are assigned the *same* exposure
> scenarios — market size (DMA rank, TV homes) is discussed as qualitative
> context only and is not used to calculate these numbers, because DMA rank
> and exposure are only weakly related across current teams (r = +0.24).
>
> **Pricing bounds** come from only **4 disclosed real jersey-patch deals**
> (Warriors, Lakers, Mavericks, 76ers) — the historical-low (0.196×), the
> observed median (0.305×), and the historical-high (0.64×) of those four
> deals' price-to-exposure ratios. A 4-deal sample is small; treat these as
> illustrative bounds from observed history, not statistically estimated
> coefficients.
>
> **Why exposure, not historical deal data, for these two cities:** no
> expansion team has signed a jersey-patch deal, so there's no historical
> price to compare. Current media-exposure data is used as the input instead,
> converted using the same historical price-to-exposure ratios observed from
> real NBA deals.

## Output 4 — How this differs from the old method

| | Old (DMA-percentile mapping) | New (two-stage scenario matrix) |
|---|---|---|
| Role of DMA rank | Mathematically selects a percentile position within a cluster's exposure distribution — directly drives the output number | Qualitative context only; **never enters a formula** |
| Why this matters | DMA rank and exposure are only weakly related (r = +0.24, p ≈ 0.21) — the old method gave market size far more predictive weight than that relationship supports | Removes that unsupported leverage entirely |
| Exposure scenarios | A single city-specific exposure estimate, implicitly precise | Three explicit scenarios (Low/Typical/High) from the actual spread of current teams' exposure — uncertainty is shown, not hidden |
| Seattle vs. Las Vegas | Produced different numbers (driven by their different DMA ranks) | Produce **identical** exposure-scenario numbers — any difference between the cities is now presented qualitatively, not manufactured numerically |
| Pricing conversion | Same 0.196×–0.64× range | Same bounds, **plus** the observed median (0.305×) shown explicitly as a third reference point, with the exact 4-deal calculation shown |
| Transparency | Percentile-mapping logic wasn't preserved anywhere in the project | Every formula, the comparison set, and the exclusion rationale are printed by the script and documented here |

## Limitations to carry onto the poster

- The old method's actual code/output could not be located in this project —
  the comparison above is based on its *description*, not a side-by-side
  numeric diff.
- The comparison set (28 teams) is a judgment call, explained above; a
  different reasonable exclusion rule would shift the percentiles somewhat
  (shown not to be large — see Stage 1 output for the all-30-team alternative).
- Four disclosed deals is a small sample for the pricing ratios; the
  historical-low/high bounds are *the full observed range of those four*, not
  a statistical confidence interval.
- These numbers assume Seattle and Las Vegas would track the broad current-
  league exposure distribution. Neither city has an NBA roster, local media
  deal, or jersey sponsor to validate that assumption against.
