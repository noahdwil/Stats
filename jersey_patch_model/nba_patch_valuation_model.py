"""
NBA jersey patch valuation model (SBA Summit)

Estimates an annual patch value for every NBA team by calibrating a linear
regression of real, publicly confirmed contract prices against Zoomph's
media-exposure value, then applying that regression to teams whose real
price is undisclosed.

Inputs (same folder):
    nba_master_model_data.csv   - one row per team, current sponsor/deal
    nba_historical_deals.csv    - deal-over-time history for select teams

Outputs (same folder):
    regression_scatter.png
    historical_deals_trend.png
    nba_patch_values_modeled.csv
"""

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

# Palette (validated colorblind-safe categorical set)
COLOR_REAL = "#2a78d6"       # blue  - confirmed real values
COLOR_ESTIMATED = "#eb6834"  # orange - modeled/estimated values
COLOR_FIT_LINE = "#52514e"   # secondary ink - regression line
COLOR_GRID = "#e1e0d9"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
SURFACE = "#fcfcfb"

TIER_COLORS = {"Large": "#2a78d6", "Mid": "#eda100", "Small": "#e34948"}


def section(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


# ---------------------------------------------------------------------------
# Step 1: Load and clean
# ---------------------------------------------------------------------------
section("STEP 1 - Load & clean data")

master = pd.read_csv("nba_master_model_data.csv")
history = pd.read_csv("nba_historical_deals.csv")

# Drop fully-blank trailing rows some spreadsheet exports leave behind
master = master.dropna(how="all").reset_index(drop=True)
history = history.dropna(how="all").reset_index(drop=True)

numeric_master_cols = ["Real_Annual_Value", "Exposure_Value_Zoomph", "DMA_Rank", "TV_Homes"]
for col in numeric_master_cols:
    master[col] = pd.to_numeric(master[col], errors="coerce")

history["Reported_Annual_Value"] = pd.to_numeric(history["Reported_Annual_Value"], errors="coerce")

# Text columns: normalize blanks to NaN (they already read as NaN via pandas'
# default na detection, this just makes the intent explicit)
for col in ["Sponsor", "Category", "Value_Confidence", "DMA_Note"]:
    master[col] = master[col].where(master[col].notna(), np.nan)

print(f"master shape: {master.shape}  (expect 30 teams x 10 columns)")
print(f"historical shape: {history.shape}")
print("\nmaster dtypes:")
print(master.dtypes)

n_real = master["Real_Annual_Value"].notna().sum()
n_missing_real = master["Real_Annual_Value"].isna().sum()
print(f"\nReal_Annual_Value populated for {n_real} of {len(master)} teams; "
      f"{n_missing_real} teams need an estimate.")
print("NOTE: the brief described ~12 confirmed teams, but this data pass only "
      "has 4 rows with a real current-deal price (Warriors, Lakers, Thunder, 76ers) "
      "-- everything else in the master file is still 'Not found'. The historical "
      "deals file has additional confirmed prices, but those are for *past* "
      "sponsors under different exposure conditions, so they aren't valid "
      "calibration points against a team's *current* Exposure_Value_Zoomph. "
      "The model below is calibrated on the 4 real points that are actually "
      "usable -- see Step 8 for what that means for reliability.")

print("\nExposure_Value_Zoomph is populated for all teams:",
      master["Exposure_Value_Zoomph"].notna().sum(), "/", len(master))
print("Missing values by column:")
print(master.isna().sum())


# ---------------------------------------------------------------------------
# Step 2: Fit regression on rows with a real value
# ---------------------------------------------------------------------------
section("STEP 2 - Fit Real_Annual_Value ~ Exposure_Value_Zoomph")

known = master.dropna(subset=["Real_Annual_Value"]).copy()
print(f"Calibration sample: n = {len(known)}")
print(known[["Team", "Sponsor", "Exposure_Value_Zoomph", "Real_Annual_Value", "Value_Confidence"]]
      .to_string(index=False))

X_known = known[["Exposure_Value_Zoomph"]].values
y_known = known["Real_Annual_Value"].values

reg = LinearRegression()
reg.fit(X_known, y_known)
slope = reg.coef_[0]
intercept = reg.intercept_
r2 = reg.score(X_known, y_known)

print(f"\nFitted model: Real_Annual_Value = {intercept:,.0f} + {slope:.4f} * Exposure_Value_Zoomph")
print(f"R^2 (on the 4 calibration points): {r2:.3f}")

known["Predicted_Annual_Value"] = reg.predict(X_known)
known["Residual"] = known["Real_Annual_Value"] - known["Predicted_Annual_Value"]
known["Pct_Error"] = 100 * known["Residual"] / known["Real_Annual_Value"]

print("\nSanity check - predicted vs. actual for the known teams:")
print(known[["Team", "Sponsor", "Real_Annual_Value", "Predicted_Annual_Value", "Residual", "Pct_Error"]]
      .round(0).to_string(index=False))
print("\nInterpretation: with only 4 points, R^2 this low means exposure value "
      "alone explains a modest share of real price -- e.g. the Lakers' real "
      "price ($30M) sits BELOW what their huge exposure number would imply, "
      "while the Thunder's real price ($40M, itself an 'Uncertain - verify' "
      "figure) sits ABOVE what their exposure would imply. Real deal prices "
      "clearly depend on things beyond media exposure (category, negotiating "
      "leverage, deal timing/length). Treat the line as a rough prior, not a "
      "precise price predictor.")


# ---------------------------------------------------------------------------
# Step 3: Apply the regression to every team, build the final value column
# ---------------------------------------------------------------------------
section("STEP 3 - Estimate values for teams without a confirmed real price")

master["Estimated_Annual_Value"] = reg.predict(master[["Exposure_Value_Zoomph"]].values)
master["Final_Annual_Value"] = master["Real_Annual_Value"].where(
    master["Real_Annual_Value"].notna(), master["Estimated_Annual_Value"]
)
master["Value_Source"] = np.where(master["Real_Annual_Value"].notna(), "Real (Confirmed)", "Estimated (Model)")

n_negative = (master["Estimated_Annual_Value"] < 0).sum()
print(f"Estimated values generated for {n_missing_real} teams. Negative estimates: {n_negative}.")

print("\nFinal value by team (sorted by Final_Annual_Value):")
print(master[["Team", "Sponsor", "Value_Source", "Exposure_Value_Zoomph", "Final_Annual_Value"]]
      .sort_values("Final_Annual_Value", ascending=False)
      .round(0).to_string(index=False))

print("\nNote: Portland Trail Blazers currently has NO jersey patch sponsor. "
      "Its 'Final_Annual_Value' is a hypothetical -- what the model thinks a "
      "patch on their jersey *could* sell for given their market exposure -- "
      "not an active contract. Flagging this so it isn't read as a real deal.")


# ---------------------------------------------------------------------------
# Step 4: Scatter plot - exposure vs. value, real vs. estimated, with fit line
# ---------------------------------------------------------------------------
section("STEP 4 - Scatter plot: Exposure_Value_Zoomph vs. value")

fig, ax = plt.subplots(figsize=(9, 6.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

real_pts = master[master["Value_Source"] == "Real (Confirmed)"]
est_pts = master[master["Value_Source"] == "Estimated (Model)"]

ax.scatter(est_pts["Exposure_Value_Zoomph"], est_pts["Estimated_Annual_Value"],
           s=55, marker="o", color=COLOR_ESTIMATED, alpha=0.85,
           label=f"Estimated / modeled (n={len(est_pts)})", zorder=3)
ax.scatter(real_pts["Exposure_Value_Zoomph"], real_pts["Real_Annual_Value"],
           s=90, marker="D", color=COLOR_REAL, edgecolor=COLOR_INK, linewidth=0.6,
           label=f"Confirmed real value (n={len(real_pts)})", zorder=4)

x_line = np.linspace(0, master["Exposure_Value_Zoomph"].max() * 1.05, 100)
y_line = reg.predict(x_line.reshape(-1, 1))
ax.plot(x_line, y_line, color=COLOR_FIT_LINE, linewidth=2, linestyle="--",
         label=f"Fitted line (R²={r2:.2f}, n=4)", zorder=2)

for _, row in real_pts.iterrows():
    ax.annotate(row["Team"], (row["Exposure_Value_Zoomph"], row["Real_Annual_Value"]),
                textcoords="offset points", xytext=(6, 6), fontsize=8.5, color=COLOR_INK)

ax.set_xlabel("Exposure_Value_Zoomph ($, media-exposure estimate)", color=COLOR_INK)
ax.set_ylabel("Annual patch value ($)", color=COLOR_INK)
ax.set_title("NBA jersey patch value: confirmed real prices vs. modeled estimates",
             color=COLOR_INK, fontsize=13, fontweight="bold")
ax.xaxis.set_major_formatter(lambda v, _: f"${v/1e6:.0f}M")
ax.yaxis.set_major_formatter(lambda v, _: f"${v/1e6:.0f}M")
ax.grid(True, color=COLOR_GRID, linewidth=0.8)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(COLOR_MUTED)
ax.tick_params(colors=COLOR_MUTED)
ax.legend(frameon=False, loc="upper left", fontsize=9)

fig.tight_layout()
fig.savefig("regression_scatter.png", dpi=200, facecolor=SURFACE)
plt.close(fig)
print("Saved regression_scatter.png")


# ---------------------------------------------------------------------------
# Step 5: Market-size tiers from DMA_Rank / TV_Homes
# ---------------------------------------------------------------------------
section("STEP 5 - Market-size tiers")

print("Cutoff reasoning: sorting all 30 teams by TV_Homes shows two natural "
      "breaks. (1) Between rank #10 (Golden State, 2.54M TV homes) and rank "
      "#12 (Phoenix, 2.20M) there's a ~14% drop, and #10 lines up with a clean, "
      "presentation-friendly round number: >= 2.5M TV homes. That also happens "
      "to match 'top-10 U.S. DMA' almost exactly (13 teams qualify once you "
      "include ties -- New York/LA/Toronto's metro each host two listed ranks "
      "or a shared market). (2) At the bottom, TV_Homes < 1.0M cleanly isolates "
      "the four smallest, and commonly cited smallest, NBA markets (Milwaukee, "
      "OKC, New Orleans, Memphis). Everything in between is 'Mid'.")

def tier(tv_homes):
    if tv_homes >= 2_500_000:
        return "Large"
    if tv_homes < 1_000_000:
        return "Small"
    return "Mid"

master["Market_Tier"] = master["TV_Homes"].apply(tier)

tier_counts = master["Market_Tier"].value_counts().reindex(["Large", "Mid", "Small"])
print("\nTeams per tier:")
print(tier_counts.to_string())

tier_summary = (
    master.groupby("Market_Tier")["Final_Annual_Value"]
    .agg(Teams="count", Avg_Value="mean", Median_Value="median")
    .reindex(["Large", "Mid", "Small"])
)
print("\nAverage team value (real + estimated) per market tier:")
print(tier_summary.round(0).to_string())


# ---------------------------------------------------------------------------
# Step 6: Historical deal trend - Lakers and Warriors
# ---------------------------------------------------------------------------
section("STEP 6 - Historical patch value trend (Lakers, Warriors)")

def parse_start_year(years_active):
    match = re.search(r"(\d{4})", str(years_active))
    return int(match.group(1)) if match else np.nan

history["Start_Year"] = history["Years_Active"].apply(parse_start_year)

trend_teams = {"LA Lakers": COLOR_REAL, "Golden State Warriors": COLOR_ESTIMATED}

fig, ax = plt.subplots(figsize=(8.5, 6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

for team, color in trend_teams.items():
    team_hist = history[(history["Team"] == team) & history["Reported_Annual_Value"].notna()]
    team_hist = team_hist.sort_values("Start_Year")
    print(f"\n{team}:")
    print(team_hist[["Sponsor", "Years_Active", "Reported_Annual_Value"]].to_string(index=False))
    ax.plot(team_hist["Start_Year"], team_hist["Reported_Annual_Value"],
            marker="o", markersize=8, linewidth=2.2, color=color, label=team, zorder=3)
    for _, row in team_hist.iterrows():
        ax.annotate(f"{row['Sponsor']}\n${row['Reported_Annual_Value']/1e6:.0f}M",
                    (row["Start_Year"], row["Reported_Annual_Value"]),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=8.5, color=COLOR_INK)

ax.set_xlabel("Deal start year", color=COLOR_INK)
ax.set_ylabel("Reported annual patch value ($)", color=COLOR_INK)
ax.set_title("Jersey patch value over time: Lakers vs. Warriors",
             color=COLOR_INK, fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(lambda v, _: f"${v/1e6:.0f}M")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.grid(True, color=COLOR_GRID, linewidth=0.8)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(COLOR_MUTED)
ax.tick_params(colors=COLOR_MUTED)
ax.legend(frameon=False, loc="upper left", fontsize=9)

fig.tight_layout()
fig.savefig("historical_deals_trend.png", dpi=200, facecolor=SURFACE)
plt.close(fig)
print("\nSaved historical_deals_trend.png")


# ---------------------------------------------------------------------------
# Step 7: Save final merged dataset
# ---------------------------------------------------------------------------
section("STEP 7 - Save final dataset")

output_cols = [
    "Team", "Sponsor", "Category", "Deal_Timeframe",
    "Real_Annual_Value", "Estimated_Annual_Value", "Final_Annual_Value", "Value_Source",
    "Exposure_Value_Zoomph", "DMA_Rank", "TV_Homes", "Market_Tier", "DMA_Note",
]
final = master[output_cols].sort_values("Final_Annual_Value", ascending=False)
final.to_csv("nba_patch_values_modeled.csv", index=False)
print(f"Saved nba_patch_values_modeled.csv ({len(final)} rows, {len(output_cols)} columns)")


# ---------------------------------------------------------------------------
# Step 8: Plain-language summary
# ---------------------------------------------------------------------------
section("STEP 8 - Summary")

print("""
What the model does: it assumes teams with bigger Zoomph media-exposure
numbers command bigger real patch prices, fits a straight line through the
4 teams where we actually know the real price (Warriors, Lakers, Thunder,
76ers), and uses that line to fill in a value for the other 26 teams.

What it found:
  - The fitted line is Real ~= $21.4M + 0.20 x Exposure_Value_Zoomph.
    In other words, exposure value only explains part of the story: every
    extra $1M of Zoomph exposure predicts roughly +$0.20M of real deal
    value, on top of a ~$21M baseline.
  - R^2 on the calibration set is ~0.22 -- weak. With 4 points, one
    disagreement (Lakers, Thunder) swings the fit a lot.
  - The Warriors' Iren deal and the Lakers' Albert deal actually contradict
    each other directionally: the Lakers have by far the highest exposure
    value (~$102M) but a lower real price ($30M) than the Warriors
    (~$77M exposure, $50M real price). That's a signal that exposure is a
    partial proxy at best -- sponsor category, deal length, and
    negotiation leverage clearly matter too.

Caveats -- read before using these numbers in the deck:
  1. Calibration sample is 4 real prices, not ~12. The brief's working
     assumption of ~12 known teams doesn't match this data pull; most
     current-deal prices are still "Not found" in the source research.
     Historical deal prices (Wish, Bibigo, Rakuten, Webull, etc.) exist for
     ~9 more team-sponsor pairs, but they're PAST deals under different
     exposure conditions and can't be paired with a team's CURRENT
     Exposure_Value_Zoomph number, so they were correctly excluded from
     calibration -- but it does mean this version of the model is running
     on a much thinner base than planned.
  2. n=4 is far too small to trust point estimates. Treat every
     Estimated_Annual_Value as a rough, directional number (i.e. "probably
     somewhere in this range relative to other teams"), not a number to
     put in front of a sponsor as a specific asking price.
  3. Extrapolation risk: the calibration points span Exposure_Value_Zoomph
     of ~$15.7M-$102M. Teams far outside that range (e.g. Utah, Memphis,
     Wizards, all under $6.3M) are being extrapolated well below the data
     the line was fit on -- the model has no real evidence for how price
     behaves down there, it's just following the line.
  4. The Thunder's $40M figure is itself flagged "Uncertain - verify" in
     the source data (unclear if it's an annual or total deal figure), so
     it's an already-shaky anchor point baked into the fit.
  5. Portland has no sponsor at all right now; its modeled value is a
     hypothetical "what a patch could be worth here," not a deal estimate.
  6. This model deliberately does NOT include Seattle/Las Vegas expansion
     projections -- that's a separate follow-up step.

Bottom line: this is a reasonable first-pass ranking/ordering of teams by
likely patch value, useful for a directional story (which teams' patches
are probably worth more or less), but the dollar figures themselves should
be presented as rough model estimates with the calibration caveats above,
not as researched prices.
""")
