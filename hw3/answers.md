# HW3 Problem 6: Benford's Law

## (a) Verify P(D=j) = log10((j+1)/j) is a valid PMF

A valid PMF requires (1) each probability is non-negative, and (2) the probabilities sum to 1.

1. **Non-negativity**: for j in {1,...,9}, (j+1)/j > 1, so log10((j+1)/j) > 0.
2. **Sums to 1**: the sum telescopes,

   sum_{j=1}^{9} log10((j+1)/j) = sum_{j=1}^{9} [log10(j+1) - log10(j)]
   = log10(10) - log10(1) = 1 - 0 = 1.

Computed values (see `benfords_law.py`):

| j | P(D=j) |
|---|--------|
| 1 | 0.3010 |
| 2 | 0.1761 |
| 3 | 0.1249 |
| 4 | 0.0969 |
| 5 | 0.0792 |
| 6 | 0.0669 |
| 7 | 0.0580 |
| 8 | 0.0512 |
| 9 | 0.0458 |

Sum = 1.0000, all non-negative. So this is a valid PMF.

## (b) Leading digit histogram

Ran `benfords_law.py` on `us_county_population_2010.csv`, extracting the leading digit of the `population_2010` column for all 3,142 counties and plotting the observed relative frequencies against Benford's Law. See `benford_histogram.png`.

Observed vs. Benford frequencies:

| Digit | Observed | Benford | Diff |
|-------|----------|---------|------|
| 1 | 0.3033 | 0.3010 | +0.0023 |
| 2 | 0.1891 | 0.1761 | +0.0130 |
| 3 | 0.1190 | 0.1249 | -0.0059 |
| 4 | 0.0980 | 0.0969 | +0.0011 |
| 5 | 0.0678 | 0.0792 | -0.0114 |
| 6 | 0.0668 | 0.0669 | -0.0001 |
| 7 | 0.0579 | 0.0580 | -0.0001 |
| 8 | 0.0484 | 0.0512 | -0.0028 |
| 9 | 0.0497 | 0.0458 | +0.0039 |

## (c) Is the plot sufficiently convincing?

The bar chart shows the observed leading-digit frequencies tracking Benford's Law closely in shape (decreasing frequency as the digit increases, largest gap between digits 1 and 2, small overall deviations), which is visually suggestive. However, a plot alone is not sufficient to *determine* conformity rigorously:

- The differences are eyeballed, with no sense of whether a deviation of, e.g., 0.013 for digit 2 is "large" relative to sampling variability.
- A formal statistical test (e.g., a chi-squared goodness-of-fit test comparing observed counts to expected counts under Benford's Law) is needed to quantify whether the deviations are within what we'd expect from natural sampling variation versus a systematic departure from Benford's Law.

So the plot is good supporting/exploratory evidence but should be paired with a formal hypothesis test before concluding the data does or does not conform to Benford's law.
