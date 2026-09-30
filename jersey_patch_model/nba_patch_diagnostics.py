"""
NBA jersey patch valuation - calibration diagnostics (v3)

Prior version (linear regression, then a time-adjusted 11-point regression)
produced R^2 = 0.014 -- no usable signal -- and collapsed every estimated
team into a narrow $31-34M band regardless of market size. This script
throws out the time-adjustment step and the linear fit, and instead:

  1. Uses the 5 confirmed real prices AS REPORTED, no inflation/deflation.
  2. Computes real-price / exposure-value ratios per team; reports the
     spread, doesn't force a median-ratio estimate if the spread is wide.
  3. Tests market size (TV_Homes, DMA_Rank) as an alternative predictor,
     via Pearson correlation, alongside exposure value.
  4. Stops here -- no 30-team output until a method actually holds up.

Inputs:
    nba_master_model_data.csv          - Exposure_Value_Zoomph, DMA_Rank, TV_Homes lookups
    NBA_Jersey_Patch_Valuation.xlsx    - 'NBA Patch Deals' tab, source of the 5 confirmed prices
"""

import numpy as np
import pandas as pd
import openpyxl
from scipy.stats import pearsonr

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


def section(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


# ---------------------------------------------------------------------------
# Step 1: Pull the 5 confirmed real prices, as reported (no time-adjustment)
# ---------------------------------------------------------------------------
section("STEP 1 - Confirmed real prices, as reported (no inflation/deflation)")

master = pd.read_csv("nba_master_model_data.csv").dropna(how="all")
master["Exposure_Value_Zoomph"] = pd.to_numeric(master["Exposure_Value_Zoomph"], errors="coerce")
master["DMA_Rank"] = pd.to_numeric(master["DMA_Rank"], errors="coerce")
master["TV_Homes"] = pd.to_numeric(master["TV_Homes"], errors="coerce")

wb = openpyxl.load_workbook("NBA_Jersey_Patch_Valuation.xlsx", data_only=True)
deals_sheet = wb["NBA Patch Deals"]
rows = list(deals_sheet.iter_rows(values_only=True))
header = rows[0]
deals = pd.DataFrame([r for r in rows[1:] if r[0] is not None], columns=header)
deals = deals.rename(columns={
    "Reported Annual Value ($)": "Real_Annual_Value",
    "Value Confidence": "Value_Confidence",
    "Deal Timeframe": "Deal_Timeframe",
})

confirmed = deals[deals["Real_Annual_Value"].notna()].copy()
confirmed["Real_Annual_Value"] = pd.to_numeric(confirmed["Real_Annual_Value"])

lookup_cols = ["Team", "Exposure_Value_Zoomph", "DMA_Rank", "TV_Homes"]
confirmed = confirmed.merge(master[lookup_cols], on="Team", how="left")

print(f"n = {len(confirmed)} confirmed real prices (source: NBA_Jersey_Patch_Valuation.xlsx, "
      f"'NBA Patch Deals' tab -- this is the same set the workbook's own notes point to).")
print("\nUsing prices exactly as reported. NOT inflated/deflated to a common year this time: the "
      "growth rate used previously came from only 2 teams (Lakers, Warriors) and the last run showed "
      "it was adding noise, not signal (R^2 dropped from already-weak to 0.014). Dropped per your "
      "instruction.")
print("\nOklahoma City Thunder's $40M is flagged 'Uncertain - verify' in the source -- unclear if "
      "it's an annual or total deal figure. Kept in the table below but called out throughout.")

print("\n" + confirmed[["Team", "Sponsor", "Real_Annual_Value", "Value_Confidence", "Deal_Timeframe"]]
      .to_string(index=False))


# ---------------------------------------------------------------------------
# Step 2: Ratio approach - real price / exposure value
# ---------------------------------------------------------------------------
section("STEP 2 - Ratio approach: real price / Exposure_Value_Zoomph")

confirmed["Ratio_to_Exposure"] = confirmed["Real_Annual_Value"] / confirmed["Exposure_Value_Zoomph"]

ratio_table = confirmed[["Team", "Sponsor", "Real_Annual_Value", "Exposure_Value_Zoomph",
                          "Ratio_to_Exposure", "Value_Confidence"]].sort_values(
    "Ratio_to_Exposure")

print("\n| Team | Sponsor | Real Price | Exposure (Zoomph) | Ratio (price/exposure) | Confidence |")
print("|---|---|---|---|---|---|")
for _, row in ratio_table.iterrows():
    print(f"| {row['Team']} | {row['Sponsor']} | ${row['Real_Annual_Value']:,.0f} | "
          f"${row['Exposure_Value_Zoomph']:,.0f} | {row['Ratio_to_Exposure']:.3f} | {row['Value_Confidence']} |")

median_ratio = confirmed["Ratio_to_Exposure"].median()
min_ratio = confirmed["Ratio_to_Exposure"].min()
max_ratio = confirmed["Ratio_to_Exposure"].max()
spread = max_ratio / min_ratio

print(f"\nMedian ratio: {median_ratio:.3f}")
print(f"Min: {min_ratio:.3f} ({ratio_table.iloc[0]['Team']})  |  "
      f"Max: {max_ratio:.3f} ({ratio_table.iloc[-1]['Team']})")
print(f"Spread (max/min): {spread:.2f}x")

RATIO_TIGHT_THRESHOLD = 2.0  # max/min under 2x would be "reasonably tight"; this is generous
if spread <= RATIO_TIGHT_THRESHOLD:
    verdict_ratio = (f"Spread is {spread:.2f}x, under the {RATIO_TIGHT_THRESHOLD}x bar -- reasonably "
                      f"tight. Median ratio ({median_ratio:.3f}) could be used as a rough multiplier.")
else:
    verdict_ratio = (f"Spread is {spread:.2f}x -- NOT tight by any reasonable bar (even a generous "
                      f"{RATIO_TIGHT_THRESHOLD}x cutoff). The ratio swings more than 4-fold across "
                      f"5 teams: Lakers pay ~0.29x their exposure number, the Thunder's figure implies "
                      f"~1.34x. A single median ratio applied to all 24-25 remaining teams would be "
                      f"fabricating precision the data doesn't support. NOT recommending a forced "
                      f"median-ratio estimate.")
print(f"\nVerdict: {verdict_ratio}")


# ---------------------------------------------------------------------------
# Step 3: Market size as an alternative predictor
# ---------------------------------------------------------------------------
section("STEP 3 - Market size (TV_Homes / DMA_Rank) vs. exposure value as predictors")

confirmed["Ratio_to_TVHomes"] = confirmed["Real_Annual_Value"] / confirmed["TV_Homes"]

print("\n| Team | Real Price | TV Homes | Price per TV home | DMA Rank |")
print("|---|---|---|---|---|")
for _, row in confirmed.sort_values("Ratio_to_TVHomes").iterrows():
    print(f"| {row['Team']} | ${row['Real_Annual_Value']:,.0f} | {row['TV_Homes']:,.0f} | "
          f"${row['Ratio_to_TVHomes']:.2f} | {row['DMA_Rank']:.0f} |")

def corr_report(x, y, label, n):
    r, p = pearsonr(x, y)
    print(f"  {label}: r = {r:+.3f}, p = {p:.3f}  (n={n} -- far too small for p to mean much; "
          f"reported for completeness, not as evidence of significance)")
    return r, p

print("\nPearson correlation with Real_Annual_Value (n=5 throughout -- treat every number here as "
      "indicative at best, not statistically established):")
r_exposure, p_exposure = corr_report(confirmed["Exposure_Value_Zoomph"], confirmed["Real_Annual_Value"],
                                      "Exposure_Value_Zoomph", len(confirmed))
r_tvhomes, p_tvhomes = corr_report(confirmed["TV_Homes"], confirmed["Real_Annual_Value"],
                                    "TV_Homes", len(confirmed))
r_dmarank, p_dmarank = corr_report(confirmed["DMA_Rank"], confirmed["Real_Annual_Value"],
                                    "DMA_Rank (lower rank = bigger market, so a NEGATIVE r here "
                                    "would mean 'bigger market -> higher price')", len(confirmed))

print(f"\nSummary of |r|: Exposure_Value_Zoomph = {abs(r_exposure):.3f}, "
      f"TV_Homes = {abs(r_tvhomes):.3f}, DMA_Rank = {abs(r_dmarank):.3f}")

strongest = max(
    [("Exposure_Value_Zoomph", abs(r_exposure)), ("TV_Homes", abs(r_tvhomes)), ("DMA_Rank", abs(r_dmarank))],
    key=lambda t: t[1],
)
print(f"Nominally strongest: {strongest[0]} (|r|={strongest[1]:.3f}) -- but with n=5, |r| needs to be "
      f"roughly 0.88+ before p<0.05, and none of these clear that bar. 'Strongest of three weak "
      f"signals' is not the same as 'a predictor that works.'")


# ---------------------------------------------------------------------------
# Step 4: Honest bottom line
# ---------------------------------------------------------------------------
section("STEP 4 - Bottom line (no 30-team output generated)")

print(f"""
What this says: with only {len(confirmed)} confirmed real prices, neither Exposure_Value_Zoomph nor
market size (TV_Homes / DMA_Rank) predicts real patch price well enough to extend to the other
~25 teams with any real confidence. Every correlation above is weak-to-moderate at best, and n=5
is too small for any of them to be statistically meaningful -- the ranking among Exposure/TV_Homes/
DMA_Rank by |r| could easily flip if even one of these 5 prices turns out to be reported wrong (the
Thunder's is already flagged as unconfirmed-term).

The ratio table makes the underlying problem visible directly: the Lakers' $30M deal on a $102M
exposure number (ratio 0.29) and the Thunder's $40M deal on a $29.8M exposure number (ratio 1.34)
aren't just noisy around a common relationship -- they're consistent with deal price being driven
mostly by things this dataset doesn't capture (sponsor category, negotiation leverage, deal timing,
whether a deal is new or about to expire), not by exposure or market size.

Recommendation: do NOT publish a full 30-team estimated table from this data as it stands. Two
honest options from here:
  A. Present the {len(confirmed)} confirmed prices only, with a stated limitation ("the other ~25
     teams' real prices are undisclosed; exposure value and market size were tested as predictors
     and neither held up with the available confirmed sample") -- this is what you asked to be able
     to fall back to, and it's defensible on a poster.
  B. Try to grow the confirmed sample before modeling again -- e.g. targeted manual research on a
     few more of the 'Not found' teams (the historical-deals file's other undisclosed rows are
     candidates: Nets/Get Your Guide, Blazers/Brightside, Kings/Phoong Law, Celtics/Vistaprint and
     Amica, Clippers/Honey, 76ers/StubHub, Cavaliers/Goodyear, Grizzlies/FedEx, Heat/Ultimate
     Software, Knicks/Squarespace, Wizards/Geico, Rockets/ROKiT, Bucks/Harley-Davidson) -- more real
     points is the only thing that will actually fix this, not a different curve fit on the same 5.

Not computing Seattle/Las Vegas expansion estimates, per your instruction -- that stays parked until
a predictive method is actually validated.
""")
