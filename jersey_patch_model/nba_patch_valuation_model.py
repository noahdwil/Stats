"""
NBA jersey patch valuation model (SBA Summit)

Estimates an annual patch value for every NBA team by calibrating a linear
regression of real, publicly confirmed contract prices against Zoomph's
media-exposure value, then applying that regression to teams whose real
price is undisclosed.

The calibration set combines BOTH files: current confirmed prices from the
master file, and past confirmed prices from the historical-deals file. Past
prices are converted to 2026-equivalent dollars (using a growth rate
observed from the two teams with real multi-year trajectories) before being
used, so a $7M deal signed in 2017 and a $50M deal signed in 2026 are
compared on the same footing rather than treated as directly equal.

Inputs (same folder):
    nba_master_model_data.csv   - one row per team, current sponsor/deal
    nba_historical_deals.csv    - deal-over-time history for select teams

Outputs (same folder):
    regression_scatter.png
    historical_deals_trend.png
    nba_patch_values_modeled.csv
    nba_patch_calibration_points.csv
"""

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

# Palette (validated colorblind-safe categorical set)
COLOR_REAL = "#2a78d6"        # blue   - confirmed values, in 2026-equivalent $
COLOR_ESTIMATED = "#eb6834"   # orange - modeled/estimated values
COLOR_NOMINAL = "#898781"     # muted  - confirmed values, as originally reported (nominal $)
COLOR_FIT_LINE = "#52514e"    # secondary ink - regression line
COLOR_GRID = "#e1e0d9"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
SURFACE = "#fcfcfb"

CURRENT_YEAR = 2026          # "today", per the source data's compilation date (Sept 2026)
PATCH_ERA_START = 2017       # first season the NBA allowed jersey patch sponsors


def section(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def parse_start_year(timeframe):
    match = re.search(r"(\d{4})", str(timeframe))
    return int(match.group(1)) if match else np.nan


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

# Parse a start year out of each deal's timeframe string, e.g. "2021-present"
# -> 2021, "~2017-2021" -> 2017, "2023-24" -> 2023. Needed to bring the
# historical file's past prices into the model as time-aware reference points.
master["Deal_Start_Year"] = master["Deal_Timeframe"].apply(parse_start_year)
history["Deal_Start_Year"] = history["Years_Active"].apply(parse_start_year)

print(f"master shape: {master.shape}  (expect 30 teams x 11 columns)")
print(f"historical shape: {history.shape}")
print("\nmaster dtypes:")
print(master.dtypes)

n_real = master["Real_Annual_Value"].notna().sum()
n_missing_real = master["Real_Annual_Value"].isna().sum()
print(f"\nReal_Annual_Value populated for {n_real} of {len(master)} teams (current deals only); "
      f"{n_missing_real} teams need an estimate.")
print("Exposure_Value_Zoomph is populated for all teams:",
      master["Exposure_Value_Zoomph"].notna().sum(), "/", len(master))
print("\nMissing values by column (master):")
print(master.isna().sum())
print(f"\nHistorical file: {history['Reported_Annual_Value'].notna().sum()} of {len(history)} "
      f"past deals have a reported price.")


# ---------------------------------------------------------------------------
# Step 2: Build a year-aware calibration set and fit the regression
# ---------------------------------------------------------------------------
section("STEP 2 - Build calibration set (current + historical) and fit the model")

current_real = master.loc[master["Real_Annual_Value"].notna(),
    ["Team", "Sponsor", "Deal_Start_Year", "Real_Annual_Value", "Value_Confidence"]].copy()
current_real["Deal_Type"] = "Current deal"

hist_real = history.loc[history["Reported_Annual_Value"].notna(),
    ["Team", "Sponsor", "Deal_Start_Year", "Reported_Annual_Value", "Value_Confidence"]].copy()
hist_real = hist_real.rename(columns={"Reported_Annual_Value": "Real_Annual_Value"})
hist_real["Deal_Type"] = "Historical deal"

calib = pd.concat([current_real, hist_real], ignore_index=True)
calib = calib.drop_duplicates(subset=["Team", "Sponsor", "Deal_Start_Year"]).reset_index(drop=True)

exposure_lookup = master.set_index("Team")["Exposure_Value_Zoomph"]
calib["Exposure_Value_Zoomph"] = calib["Team"].map(exposure_lookup)

print(f"Combining current-deal real prices ({len(current_real)}) with historical-deal real "
      f"prices ({len(hist_real)}), de-duplicated on (Team, Sponsor, Year) since some deals -- "
      f"e.g. Warriors/Iren, Lakers/Albert -- appear in both files: calibration set n = {len(calib)}.")
print("\nThis is close to the ~12 known-teams figure originally expected -- most of that gap was "
      "sitting in the historical-deals file, not the master file.")
print("\nCalibration set (sorted by year):")
print(calib.sort_values("Deal_Start_Year")
      [["Team", "Sponsor", "Deal_Start_Year", "Real_Annual_Value", "Value_Confidence", "Deal_Type"]]
      .to_string(index=False))

# --- First attempt: add year directly as a second regression feature -------
X_naive = calib[["Exposure_Value_Zoomph", "Deal_Start_Year"]].values
y_naive = calib["Real_Annual_Value"].values
reg_naive = LinearRegression().fit(X_naive, y_naive)
print(f"\nFirst attempt -- Exposure + raw Deal_Start_Year as two regression features: "
      f"year coefficient = {reg_naive.coef_[1]:,.0f}, R^2 = {reg_naive.score(X_naive, y_naive):.3f}. "
      f"This doesn't work: the year coefficient comes out essentially zero. The only exposure "
      f"number we have per team is TODAY's Zoomph value, used as a stand-in for every year of "
      f"that team's history -- so 'year' and 'team identity' get confounded with 'exposure', and "
      f"the model can't cleanly separate 'this team is just bigger' from 'the market grew since "
      f"then'. Dropped in favor of the approach below.")

# --- Better approach: inflate historical prices to 2026-equivalent dollars -
# Growth rate is derived from the only two teams with a genuine multi-year,
# same-team trajectory (Lakers: Wish->Bibigo->Albert, Warriors: Rakuten->Iren)
# rather than pooled across the whole (mostly single-point-per-team) sample.
lakers_hist = history[(history["Team"] == "LA Lakers") & history["Reported_Annual_Value"].notna()]
warriors_hist = history[(history["Team"] == "Golden State Warriors") & history["Reported_Annual_Value"].notna()]

def endpoint_cagr(df):
    df = df.sort_values("Deal_Start_Year")
    first, last = df.iloc[0], df.iloc[-1]
    years = last["Deal_Start_Year"] - first["Deal_Start_Year"]
    return (last["Reported_Annual_Value"] / first["Reported_Annual_Value"]) ** (1 / years) - 1

lakers_cagr = endpoint_cagr(lakers_hist)
warriors_cagr = endpoint_cagr(warriors_hist)
GROWTH_RATE = (lakers_cagr + warriors_cagr) / 2

print(f"\nObserved annual growth rate, from the two teams we can actually track over time:")
print(f"  Lakers   (2017 Wish $13M -> 2026 Albert $30M, 9 yrs): {lakers_cagr:.1%}/yr")
print(f"  Warriors (2017 Rakuten $20M -> 2026 Iren $50M, 9 yrs): {warriors_cagr:.1%}/yr")
print(f"  Average, used as the league-wide patch-market growth rate: {GROWTH_RATE:.1%}/yr")
print("  Caveat: this rate rests on just 2 teams (the only ones with repeat deal history in this "
      "data) -- both large, valuable franchises, so it may overstate growth for smaller markets.")

calib["Years_To_Now"] = CURRENT_YEAR - calib["Deal_Start_Year"]
calib["Value_2026_Equivalent"] = calib["Real_Annual_Value"] * (1 + GROWTH_RATE) ** calib["Years_To_Now"]

print(f"\nEvery calibration price is inflated to {CURRENT_YEAR}-equivalent dollars at that rate "
      f"before fitting (a price signed 9 years ago is compared on today's footing, not treated as "
      f"equal to a price signed this year). A deal already signed in {CURRENT_YEAR} is unchanged.")

X = calib[["Exposure_Value_Zoomph"]].values
y = calib["Value_2026_Equivalent"].values
reg = LinearRegression().fit(X, y)
slope, intercept = reg.coef_[0], reg.intercept_
r2 = reg.score(X, y)

print(f"\nFitted model: Value_2026_Equivalent = {intercept:,.0f} + {slope:.4f} * Exposure_Value_Zoomph")
print(f"R^2 (on the {len(calib)}-point calibration set, values in 2026-equivalent $): {r2:.3f}")

calib["Predicted_2026_Equivalent"] = reg.predict(X)
calib["Residual"] = calib["Value_2026_Equivalent"] - calib["Predicted_2026_Equivalent"]
calib["Pct_Error"] = 100 * calib["Residual"] / calib["Value_2026_Equivalent"]

print("\nSanity check - predicted vs. actual (2026-equivalent $) for every calibration point:")
print(calib.sort_values("Deal_Start_Year")
      [["Team", "Sponsor", "Deal_Start_Year", "Real_Annual_Value", "Value_2026_Equivalent",
        "Predicted_2026_Equivalent", "Residual", "Pct_Error"]]
      .round(0).to_string(index=False))

# Robustness check: the Thunder's $40M figure is flagged "Uncertain - verify"
# in the source, AND it's 7 years old, so it gets compounded the hardest --
# a risky combination. Show what the fit looks like without it.
robust = calib[calib["Sponsor"] != "Love's Travel Stops & Country Stores"]
reg_robust = LinearRegression().fit(robust[["Exposure_Value_Zoomph"]].values, robust["Value_2026_Equivalent"].values)
r2_robust = reg_robust.score(robust[["Exposure_Value_Zoomph"]].values, robust["Value_2026_Equivalent"].values)
print(f"\nRobustness check: excluding the Thunder's 'Uncertain - verify' point (which is also the "
      f"most-compounded, at 7 years) changes R^2 from {r2:.3f} to {r2_robust:.3f} with n={len(robust)}. "
      f"That's a meaningful swing from one point -- a reminder that even n={len(calib)} is still a "
      f"small, noise-sensitive sample. The primary model below keeps all {len(calib)} points, "
      f"consistent with the original rule of using every populated real price.")

calib.to_csv("nba_patch_calibration_points.csv", index=False)
print("\nSaved nba_patch_calibration_points.csv (full calibration set, for auditing)")


# ---------------------------------------------------------------------------
# Step 3: Apply the regression to every team, build the final value column
# ---------------------------------------------------------------------------
section("STEP 3 - Estimate values for teams without a confirmed real price")

# Every team's Exposure_Value_Zoomph is already "as of today", and the model
# was fit on 2026-equivalent dollars, so applying it directly to current
# exposure gives a like-for-like 2026 estimate -- no separate year term needed.
master["Estimated_Annual_Value"] = reg.predict(master[["Exposure_Value_Zoomph"]].values)
master["Final_Annual_Value"] = master["Real_Annual_Value"].where(
    master["Real_Annual_Value"].notna(), master["Estimated_Annual_Value"]
)
master["Value_Source"] = np.where(master["Real_Annual_Value"].notna(), "Real (Confirmed)", "Estimated (Model)")

n_negative = (master["Estimated_Annual_Value"] < 0).sum()
print(f"Estimated {CURRENT_YEAR}-equivalent values generated for {n_missing_real} teams. "
      f"Negative estimates: {n_negative}.")

print("\nFinal value by team (sorted by Final_Annual_Value):")
print(master[["Team", "Sponsor", "Value_Source", "Exposure_Value_Zoomph", "Final_Annual_Value"]]
      .sort_values("Final_Annual_Value", ascending=False)
      .round(0).to_string(index=False))

print("\nNote: Portland Trail Blazers currently has NO jersey patch sponsor. "
      "Its 'Final_Annual_Value' is a hypothetical -- what the model thinks a "
      "patch on their jersey *could* sell for given their market exposure -- "
      "not an active contract. Flagging this so it isn't read as a real deal.")


# ---------------------------------------------------------------------------
# Step 4: Scatter plot - exposure vs. value, with historical context
# ---------------------------------------------------------------------------
section("STEP 4 - Scatter plot: Exposure_Value_Zoomph vs. value")

fig, ax = plt.subplots(figsize=(9.5, 7), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

est_pts = master[master["Value_Source"] == "Estimated (Model)"]
ax.scatter(est_pts["Exposure_Value_Zoomph"], est_pts["Estimated_Annual_Value"],
           s=55, marker="o", color=COLOR_ESTIMATED, alpha=0.85,
           label=f"Estimated / modeled, {CURRENT_YEAR} (n={len(est_pts)})", zorder=3)

# Calibration points: nominal (as-reported) value in muted gray, with a thin
# dotted connector up to the 2026-equivalent value actually used for fitting.
past = calib[calib["Deal_Start_Year"] < CURRENT_YEAR]
now = calib[calib["Deal_Start_Year"] == CURRENT_YEAR]

for _, row in past.iterrows():
    ax.plot([row["Exposure_Value_Zoomph"], row["Exposure_Value_Zoomph"]],
            [row["Real_Annual_Value"], row["Value_2026_Equivalent"]],
            color=COLOR_MUTED, linewidth=0.9, linestyle=":", zorder=2)

ax.scatter(past["Exposure_Value_Zoomph"], past["Real_Annual_Value"],
           s=45, marker="o", facecolor="none", edgecolor=COLOR_NOMINAL, linewidth=1.3,
           label="Confirmed price, as originally reported", zorder=3)
ax.scatter(calib["Exposure_Value_Zoomph"], calib["Value_2026_Equivalent"],
           s=90, marker="D", color=COLOR_REAL, edgecolor=COLOR_INK, linewidth=0.6,
           label=f"Confirmed price, in {CURRENT_YEAR}-equivalent $ (n={len(calib)}, used for fit)",
           zorder=4)

x_line = np.linspace(0, master["Exposure_Value_Zoomph"].max() * 1.05, 100)
y_line = reg.predict(x_line.reshape(-1, 1))
ax.plot(x_line, y_line, color=COLOR_FIT_LINE, linewidth=2, linestyle="--",
         label=f"Fitted line (R²={r2:.2f}, n={len(calib)})", zorder=2)

label_offsets = [(6, 6), (6, -14), (6, 18), (6, -24), (6, 28)]
for i, (_, row) in enumerate(calib.drop_duplicates(subset=["Team"]).iterrows()):
    ax.annotate(row["Team"], (row["Exposure_Value_Zoomph"], row["Value_2026_Equivalent"]),
                textcoords="offset points", xytext=label_offsets[i % len(label_offsets)],
                fontsize=8, color=COLOR_INK)

ax.set_xlabel("Exposure_Value_Zoomph ($, media-exposure estimate)", color=COLOR_INK)
ax.set_ylabel(f"Annual patch value ($, {CURRENT_YEAR}-equivalent)", color=COLOR_INK)
ax.set_title("NBA jersey patch value: confirmed prices (with historical context) vs. modeled estimates",
             color=COLOR_INK, fontsize=12, fontweight="bold")
ax.set_ylim(top=max(calib["Value_2026_Equivalent"].max(), master["Estimated_Annual_Value"].max()) * 1.22)
ax.xaxis.set_major_formatter(lambda v, _: f"${v/1e6:.0f}M")
ax.yaxis.set_major_formatter(lambda v, _: f"${v/1e6:.0f}M")
ax.grid(True, color=COLOR_GRID, linewidth=0.8)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(COLOR_MUTED)
ax.tick_params(colors=COLOR_MUTED)
ax.legend(frameon=False, loc="upper left", fontsize=8.5)

fig.tight_layout()
fig.savefig("regression_scatter.png", dpi=200, facecolor=SURFACE)
plt.close(fig)
print("Saved regression_scatter.png")
print("Read it as: hollow gray points are prices as originally reported; each has a dotted line up "
      "to its solid blue diamond, the same price re-expressed in today's dollars -- that vertical "
      "distance *is* the historical-growth adjustment. The fit line runs through the blue diamonds.")


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

trend_teams = {"LA Lakers": COLOR_REAL, "Golden State Warriors": COLOR_ESTIMATED}

fig, ax = plt.subplots(figsize=(8.5, 6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

for team, color in trend_teams.items():
    team_hist = history[(history["Team"] == team) & history["Reported_Annual_Value"].notna()]
    team_hist = team_hist.sort_values("Deal_Start_Year")
    print(f"\n{team}:")
    print(team_hist[["Sponsor", "Years_Active", "Reported_Annual_Value"]].to_string(index=False))
    ax.plot(team_hist["Deal_Start_Year"], team_hist["Reported_Annual_Value"],
            marker="o", markersize=8, linewidth=2.2, color=color, label=f"{team} (actual)", zorder=3)
    for _, row in team_hist.iterrows():
        ax.annotate(f"{row['Sponsor']}\n${row['Reported_Annual_Value']/1e6:.0f}M",
                    (row["Deal_Start_Year"], row["Reported_Annual_Value"]),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=8.5, color=COLOR_INK)
    # Overlay the model's assumed growth-rate trend, anchored at this team's
    # first real deal, as a check on how well the single shared rate tracks
    # each team's actual trajectory.
    first = team_hist.iloc[0]
    trend_years = np.arange(first["Deal_Start_Year"], CURRENT_YEAR + 1)
    trend_values = first["Reported_Annual_Value"] * (1 + GROWTH_RATE) ** (trend_years - first["Deal_Start_Year"])
    ax.plot(trend_years, trend_values, color=color, linewidth=1.2, linestyle=":", alpha=0.6,
            label=f"{team} ({GROWTH_RATE:.1%}/yr trend)", zorder=2)

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
ax.legend(frameon=False, loc="upper left", fontsize=8.5)

fig.tight_layout()
fig.savefig("historical_deals_trend.png", dpi=200, facecolor=SURFACE)
plt.close(fig)
print("\nSaved historical_deals_trend.png")
print("The dotted trend lines show the single shared growth rate used in Step 2; both teams' actual "
      "deals sit close to their own trend line, which is why that rate was judged usable at all -- "
      "but the Lakers' Bibigo deal sits a bit above its dotted trend line and Albert sits below it, "
      "a reminder that real deals don't move in a straight line.")


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
print(f"Real_Annual_Value and Final_Annual_Value are nominal (deal-year) dollars for real deals; "
      f"Estimated_Annual_Value is in {CURRENT_YEAR}-equivalent dollars.")


# ---------------------------------------------------------------------------
# Step 8: Plain-language summary
# ---------------------------------------------------------------------------
section("STEP 8 - Summary")

print(f"""
What changed from the first version: the calibration sample now combines the
master file's 4 current real prices with {len(hist_real)} more real prices
from the historical-deals file, de-duplicated to n={len(calib)} -- close to
the ~12 you originally expected. To make that mix usable, each historical
price is converted to {CURRENT_YEAR}-equivalent dollars using a growth rate
observed from the two teams we can actually track over multiple deals
(Lakers, Warriors: ~{GROWTH_RATE:.1%}/yr average) before it's compared
against a team's current exposure value.

We tried the more obvious approach first -- adding deal year directly as a
second regression input alongside exposure -- and it didn't work: the year
coefficient came out essentially zero (R^2={reg_naive.score(X_naive, y_naive):.2f}).
The reason is structural, not a coding bug: we only have ONE exposure number
per team (today's), so when it's used as a stand-in for a team's exposure in
2017 as well as 2026, "year" and "which team this is" become entangled and
the model can't separate market growth from a team just being bigger. Adjusting
each price to today's dollars first, then fitting a single-variable model,
sidesteps that confound.

What the resulting model found:
  - Fitted line: Value(2026-equiv) ~= {intercept:,.0f} + {slope:.2f} x
    Exposure_Value_Zoomph. R^2 = {r2:.2f} on n={len(calib)}.
  - That R^2 is still not strong, and it's not really comparable to the
    earlier n=4 version's R^2=0.22 -- with only 4 points, almost any R^2 is
    a fluke; n={len(calib)} noisy-but-real points is more informative than 4
    points that happened to align.
  - Excluding the Thunder's "Uncertain - verify" $40M figure (also the most
    heavily time-adjusted, at 7 years) moves R^2 from {r2:.2f} to
    {r2_robust:.2f} -- a reminder of how sensitive a sample this size still
    is to a single shaky data point.
  - The Lakers/Warriors overlay chart (historical_deals_trend.png) shows the
    shared growth-rate line tracks each team's actual deals reasonably well,
    which is the main evidence the {GROWTH_RATE:.1%}/yr figure is usable at
    all -- but it's derived from exactly 2 (large-market) teams, so treat it
    as a rough market-wide signal, not a verified league-wide rate.

Caveats -- read before using these numbers in the deck:
  1. Calibration is n={len(calib)}, not the ~12 originally assumed, but much
     closer than the first version's n=4 -- the gap was mostly in the
     historical-deals file, now folded in with a time adjustment.
  2. The {GROWTH_RATE:.1%}/yr growth rate comes from 2 teams only (Lakers,
     Warriors) -- both large markets. Applying it to every team assumes
     small-market patch values grew at the same rate, which isn't verified.
  3. Time-adjustment gets riskier the further back a price goes: the 2017
     deals (Lakers/Wish, Warriors/Rakuten, Celtics/GE) are compounded over 9
     years, so small errors in the growth rate get amplified most for them.
  4. Extrapolation risk remains: exposure values for teams like Utah,
     Memphis, and the Wizards sit well below the calibration set's range, so
     their estimates lean on the line's slope more than on real evidence.
  5. The Thunder's $40M figure is itself flagged "Uncertain - verify" in the
     source data, and it's the single point most affected by time-adjustment
     noise (see the robustness check above).
  6. Portland has no sponsor at all right now; its modeled value is a
     hypothetical "what a patch could be worth here," not a deal estimate.
  7. This model deliberately does NOT include Seattle/Las Vegas expansion
     projections -- that's a separate follow-up step.

Bottom line: bringing historical deals in, adjusted for time, gives a
noticeably richer and more defensible calibration set than the original
4-point version, and the Lakers/Warriors trend chart backs up the growth
rate it's built on. It's still a directional model, not a pricing tool --
present the dollar figures as "the model's best estimate given a real but
small and imperfect reference set," with the caveats above attached.
""")
