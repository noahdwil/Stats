"""
Seattle / Las Vegas expansion-team jersey patch valuation - two-stage
scenario matrix (replaces the DMA-percentile-mapping method).

STAGE 1 (exposure scenarios): Low/Typical/High = the 25th/50th/75th
percentile of Exposure_Value_Zoomph across a comparison set of CURRENT
NBA teams. DMA rank is NOT used to compute these numbers -- see the
"Why DMA rank is not used numerically" note below. Seattle and Las Vegas
therefore receive IDENTICAL exposure-scenario dollar figures; only the
qualitative discussion differs between the two cities.

STAGE 2 (pricing conversion): each exposure scenario is converted to a
patch-value range using three HISTORICALLY OBSERVED ratios (not
statistical confidence bounds): the lowest and highest of the four
disclosed real-deal ratios (0.196x, 0.64x), plus the median of all four.

No prior expansion-team script or notebook was found anywhere in this
project (checked both CSVs, the Excel workbook's cell formulas, every
existing .py file, and the repository's other git branch) despite a
request to inspect and revise the existing method. The "old method" is
reconstructed here only from the user's own written description, not
from executable code -- flagged explicitly in the printed output and in
expansion_methodology_notes.md.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

pd.set_option("display.width", 140)


def section(title):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


# ---------------------------------------------------------------------------
# Inputs carried over from the existing project (unchanged)
# ---------------------------------------------------------------------------
master = pd.read_csv("nba_master_model_data.csv").dropna(how="all")
master["Exposure_Value_Zoomph"] = pd.to_numeric(master["Exposure_Value_Zoomph"], errors="coerce")

# Seattle / Las Vegas DMA context, from NBA_Jersey_Patch_Valuation.xlsx
# ("Market Size (DMA)" tab) and confirmed against the attached Nielsen
# 2024-25 DMA ranking PDF. QUALITATIVE CONTEXT ONLY -- not used in any
# formula below.
CITY_CONTEXT = {
    "Seattle": {
        "dma_rank": 13,
        "tv_homes": 2_098_240,
        "note": "Larger TV market than 15+ current NBA cities (incl. Detroit, Denver, Miami, "
                "Portland, San Antonio, OKC, Memphis). Former NBA market (SuperSonics, 1967-2008).",
    },
    "Las Vegas": {
        "dma_rank": 40,
        "tv_homes": 896_460,
        "note": "Smaller TV market than every current NBA city except OKC, New Orleans, Memphis. "
                "Established pro sports fandom via the Raiders (NFL) and Golden Knights (NHL).",
    },
}

# The four disclosed real-deal ratios this project has (price / exposure).
# Treated as historical observed conversions, not confidence bounds.
OBSERVED_RATIOS = {"Warriors": 0.26, "Lakers": 0.196, "Mavericks": 0.35, "76ers": 0.64}
RATIO_LOW = min(OBSERVED_RATIOS.values())
RATIO_HIGH = max(OBSERVED_RATIOS.values())


# ---------------------------------------------------------------------------
# What the old method did, and why it isn't used here
# ---------------------------------------------------------------------------
section("BEFORE STARTING - old-method inspection")
print("""
Searched for an existing expansion-team script/notebook/spreadsheet in this
project: nba_master_model_data.csv, nba_historical_deals.csv,
NBA_Jersey_Patch_Valuation.xlsx (including cell formulas, not just cached
values), every .py file in jersey_patch_model/, and the repository's other
git branch (claude/new-session-a6rmx6, which only contains an unrelated
homework assignment). NONE of them contain Seattle/Las Vegas valuation
numbers or a percentile-mapping calculation. The old method is therefore
reconstructed here ONLY from the user's own written description:
  "maps each city's DMA percentile onto each cluster's exposure
   distribution, then multiplies that exposure by the observed
   price-to-exposure range of 0.196x-0.64x"
No old-method numbers are reproduced or altered by this script, because
none exist in the project to find. This is flagged, not assumed away.

Why that method is replaced: DMA rank/TV-homes and Exposure_Value_Zoomph
are only weakly related across the current 30 teams (r = +0.24, p = 0.21
per the user's own diagnostic). Using a city's DMA percentile to select a
point within a cluster's exposure distribution therefore lets market size
drive the output far more than the data actually support -- the r=0.24
relationship doesn't justify picking a specific percentile with any
precision. The new method below removes DMA rank from the math entirely.
""")


# ---------------------------------------------------------------------------
# Stage 1: exposure scenarios from a comparison set of current teams
# ---------------------------------------------------------------------------
section("STAGE 1 - Exposure scenarios (comparison-set percentiles)")

EXCLUDED_TEAMS = ["LA Lakers", "Golden State Warriors"]
comparison_set = master[~master["Team"].isin(EXCLUDED_TEAMS)].copy()
exposure_values = comparison_set["Exposure_Value_Zoomph"].dropna()

print(f"""Comparison set: all current NBA teams EXCEPT {', '.join(EXCLUDED_TEAMS)} (n={len(exposure_values)}).
Rationale: Lakers/Warriors' exposure reflects decades of global-brand history
that no expansion team starts with in year one; including them would pull
the percentile benchmarks up based on two extreme outliers that aren't
representative of what a new franchise could plausibly open with. Every
other current team, across every performance/exposure tier, is included --
no further cherry-picking of "comparable" cities or clusters.

DMA rank is NOT part of this calculation. Low/Typical/High come only from
the distribution of Exposure_Value_Zoomph across the comparison set.""")

p25, p50, p75 = np.percentile(exposure_values, [25, 50, 75])
EXPOSURE_SCENARIOS = {"Low (P25)": p25, "Typical (P50)": p50, "High (P75)": p75}

print("\nFormula: numpy.percentile(comparison_set Exposure_Value_Zoomph, [25, 50, 75])")
for label, val in EXPOSURE_SCENARIOS.items():
    print(f"  {label}: ${val:,.0f}")

print("\nBecause this calculation uses only the comparison set's exposure distribution, "
      "Seattle and Las Vegas receive the SAME three exposure-scenario dollar figures. "
      "DMA rank and TV homes (printed below) are shown as qualitative city context only.")
for city, ctx in CITY_CONTEXT.items():
    print(f"  {city}: DMA rank {ctx['dma_rank']}, {ctx['tv_homes']:,} TV homes -- {ctx['note']}")


# ---------------------------------------------------------------------------
# Stage 2: pricing conversion using the four disclosed ratios
# ---------------------------------------------------------------------------
section("STAGE 2 - Pricing conversion (historical observed ratios)")

sorted_ratios = sorted(OBSERVED_RATIOS.values())
median_ratio = (sorted_ratios[1] + sorted_ratios[2]) / 2

print("The four disclosed real-deal ratios (price / exposure), as given:")
for team, r in OBSERVED_RATIOS.items():
    print(f"  {team}: {r}")
print(f"\nSorted: {sorted_ratios}")
print(f"Median (n=4, even -> mean of the two middle values):")
print(f"  ({sorted_ratios[1]} + {sorted_ratios[2]}) / 2 = {median_ratio}")
print(f"\nHistorical low  = exposure x {RATIO_LOW}")
print(f"Observed median = exposure x {median_ratio}")
print(f"Historical high = exposure x {RATIO_HIGH}")
print("\nThese are historical observed conversions from 4 disclosed deals, NOT statistically "
      "estimated coefficients and NOT confidence bounds.")


# ---------------------------------------------------------------------------
# Output 1: calculation table
# ---------------------------------------------------------------------------
section("OUTPUT 1 - Calculation table")

rows = []
for city in ["Seattle", "Las Vegas"]:
    for scenario_label, exposure in EXPOSURE_SCENARIOS.items():
        rows.append({
            "City": city,
            "Exposure scenario": scenario_label,
            "Exposure assumption ($)": exposure,
            "0.196x valuation ($)": exposure * RATIO_LOW,
            "Observed-median valuation ($)": exposure * median_ratio,
            "0.64x valuation ($)": exposure * RATIO_HIGH,
        })

table = pd.DataFrame(rows)
with pd.option_context("display.float_format", lambda v: f"{v:,.0f}"):
    print(table.to_string(index=False))

table.to_csv("expansion_scenario_table.csv", index=False)
print("\nSaved expansion_scenario_table.csv")


# ---------------------------------------------------------------------------
# Output 2: visualization
# ---------------------------------------------------------------------------
section("OUTPUT 2 - Visualization")

SEATTLE_COLOR = "#2F6FA8"    # reuses the project's "Middle of the Pack" blue family
VEGAS_COLOR = "#C46FB5"      # reuses the project's "Off the Radar" magenta family
GRID_COLOR = "#e1e0d9"
INK = "#0b0b0b"
MUTED = "#52514e"
AXIS_COLOR = "#898781"

scenario_order = ["Low (P25)", "Typical (P50)", "High (P75)"]
fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=200, facecolor="white", sharex=True, sharey=True)

for ax, city, color in zip(axes, ["Seattle", "Las Vegas"], [SEATTLE_COLOR, VEGAS_COLOR]):
    ax.set_facecolor("white")
    for i, scenario_label in enumerate(scenario_order):
        exposure = EXPOSURE_SCENARIOS[scenario_label]
        low_val = exposure * RATIO_LOW
        high_val = exposure * RATIO_HIGH
        mid_val = exposure * median_ratio
        y = len(scenario_order) - 1 - i

        ax.plot([low_val, high_val], [y, y], color=color, alpha=0.35, linewidth=22,
                solid_capstyle="round", zorder=2)
        ax.plot(mid_val, y, marker="D", markersize=11, color=color,
                markeredgecolor="white", markeredgewidth=1.2, zorder=4)
        ax.plot(low_val, y, marker="|", markersize=20, color=color, markeredgewidth=2, zorder=3)
        ax.plot(high_val, y, marker="|", markersize=20, color=color, markeredgewidth=2, zorder=3)

        ax.annotate(f"${low_val/1e6:.1f}M", (low_val, y), textcoords="offset points",
                    xytext=(0, -20), ha="center", fontsize=8.5, color=MUTED)
        ax.annotate(f"${mid_val/1e6:.1f}M", (mid_val, y), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9, color=INK, fontweight="bold")
        ax.annotate(f"${high_val/1e6:.1f}M", (high_val, y), textcoords="offset points",
                    xytext=(0, -20), ha="center", fontsize=8.5, color=MUTED)

    ax.set_yticks(range(len(scenario_order)))
    ax.set_yticklabels(scenario_order[::-1], fontsize=11)
    ax.set_ylim(-0.6, len(scenario_order) - 0.4)

    ctx = CITY_CONTEXT[city]
    ax.set_title(f"{city}", fontsize=16, fontweight="bold", color=color, pad=14)
    ax.text(0.5, 1.085,
            f"DMA rank {ctx['dma_rank']} · {ctx['tv_homes']:,} TV homes (context only, not used in the math)",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=MUTED, style="italic")

    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS_COLOR)
    ax.tick_params(colors=MUTED)
    ax.set_xlabel("Illustrative jersey patch value ($ millions / year)", fontsize=10.5, color=INK)
    ax.xaxis.set_major_formatter(lambda v, _: f"${v/1e6:.1f}M")

legend_handles = [
    Line2D([0], [0], marker="D", linestyle="None", markersize=9, color=MUTED,
           label="Observed-median scenario (exposure × 0.305)"),
    Line2D([0], [0], marker="|", linestyle="None", markersize=14, markeredgewidth=2, color=MUTED,
           label="Historical-low / historical-high bound (×0.196 / ×0.64)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False, fontsize=9.5,
           bbox_to_anchor=(0.5, 0.005))

fig.subplots_adjust(top=0.78, bottom=0.17, left=0.08, right=0.97, wspace=0.12)
fig.text(0.5, 0.965, "Illustrative Expansion-Team Patch Value Scenarios — Seattle & Las Vegas",
          fontsize=19, fontweight="bold", color=INK, ha="center")
fig.text(0.5, 0.925,
          "Conditional scenarios, not forecasts. Expansion has not been finalized. Exposure scenarios use the "
          "25th/50th/75th percentile of current teams' exposure (DMA rank is not used to calculate them).\n"
          "Pricing bounds come from only 4 disclosed real deals. Current exposure is compared with historical "
          "deal prices because historical exposure data for these deals were unavailable.",
          fontsize=9.5, color=MUTED, ha="center", linespacing=1.5)

fig.savefig("expansion_scenario_chart.png", facecolor="white")
plt.close(fig)
print("Saved expansion_scenario_chart.png")
