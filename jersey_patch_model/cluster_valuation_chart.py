"""
NBA jersey patch valuation - range chart by K-means cluster.

For every team, draws a range bar (exposure x observed ratio bounds from
confirmed deals), a diamond at the confirmed real price where one exists,
and -- for clusters where a confirmed-deal multiplier can be computed -- a
small dot at exposure x multiplier for the teams that don't have a
confirmed price yet. Clusters with zero confirmed deals get a range only;
no multiplier is invented for them.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Data: team -> (exposure_$M, confirmed_real_price_$M or None)
# ---------------------------------------------------------------------------
data = {
    "LA Lakers": (102.0, 20.0),
    "Golden State Warriors": (77.3, 20.0),
    "San Antonio Spurs": (75.5, None),
    "Boston Celtics": (41.0, None),
    "Minnesota Timberwolves": (35.0, None),
    "Denver Nuggets": (34.6, None),
    "Houston Rockets": (34.3, None),
    "OKC Thunder": (29.8, None),
    "New York Knicks": (26.4, None),
    "Detroit Pistons": (21.5, None),
    "Cleveland Cavaliers": (20.5, None),
    "Dallas Mavericks": (31.1, 11.0),
    "Phoenix Suns": (29.9, None),
    "Chicago Bulls": (28.3, None),
    "LA Clippers": (26.3, None),
    "Orlando Magic": (24.8, None),
    "Miami Heat": (23.2, None),
    "Milwaukee Bucks": (17.7, None),
    "Philadelphia 76ers": (15.7, 10.0),
    "Atlanta Hawks": (14.5, None),
    "Portland Trail Blazers": (14.1, None),
    "Toronto Raptors": (11.9, None),
    "Indiana Pacers": (16.8, None),
    "Brooklyn Nets": (14.1, None),
    "Charlotte Hornets": (12.2, None),
    "New Orleans Pelicans": (10.2, None),
    "Sacramento Kings": (8.25, None),
    "Washington Wizards": (6.29, None),
    "Memphis Grizzlies": (5.38, None),
    "Utah Jazz": (5.19, None),
}

CLUSTERS = [
    # Membership synced to nba_kmeans_metro_pop.py (k=4), which now uses
    # metro population in Exposure AND a 3-season-average win% in Team
    # Strength. Latest moves from that win% change: Detroit Pistons
    # Contenders -> Middle of the Pack, Dallas Mavericks Middle of the Pack
    # -> Contenders, Indiana Pacers Off the Radar -> Middle of the Pack.
    ("Household Names", "#B23A48",
     ["LA Lakers", "Golden State Warriors", "New York Knicks"]),
    ("Contenders", "#1F9FB5",
     ["Houston Rockets", "San Antonio Spurs", "Boston Celtics", "Cleveland Cavaliers",
      "OKC Thunder", "Minnesota Timberwolves", "Denver Nuggets", "Dallas Mavericks", "LA Clippers"]),
    ("Middle of the Pack", "#2F6FA8",
     ["Detroit Pistons", "Chicago Bulls", "Miami Heat",
      "Milwaukee Bucks", "Philadelphia 76ers", "Phoenix Suns", "Orlando Magic", "Atlanta Hawks",
      "Portland Trail Blazers", "Toronto Raptors", "Brooklyn Nets", "Indiana Pacers"]),
    ("Off the Radar", "#C46FB5",
     ["Sacramento Kings", "Memphis Grizzlies",
      "Charlotte Hornets", "New Orleans Pelicans", "Washington Wizards", "Utah Jazz"]),
]

RATIO_LOW, RATIO_HIGH = 0.196, 0.64  # Lakers $20M/$102.0M ; 76ers $10M/$15.7M

NEW_DEALS = [
    ("LA Lakers", 30.0, "Albert"),
    ("Golden State Warriors", 50.0, "Iren Ltd."),
    ("Philadelphia 76ers", 30.0, "Bloom Energy"),
]

# ---------------------------------------------------------------------------
# Per-cluster multiplier: mean(confirmed price / exposure) over that
# cluster's confirmed teams only. None if the cluster has no confirmed deal.
# ---------------------------------------------------------------------------
print("Cluster multipliers (mean of confirmed price / exposure):")
cluster_multiplier = {}
for name, color, teams in CLUSTERS:
    ratios = [data[t][1] / data[t][0] for t in teams if data[t][1] is not None]
    if ratios:
        mult = float(np.mean(ratios))
        cluster_multiplier[name] = mult
        detail = ", ".join(f"{t}={data[t][1]/data[t][0]:.3f}" for t in teams if data[t][1] is not None)
        print(f"  {name}: x{mult:.3f}  (from {detail})")
    else:
        cluster_multiplier[name] = None
        print(f"  {name}: no confirmed deals -> no multiplier")

# ---------------------------------------------------------------------------
# Build the plotting order: clusters top-to-bottom as given, teams within
# each cluster sorted by exposure descending.
# ---------------------------------------------------------------------------
ordered_teams = []   # (team, cluster_name, color)
cluster_boundaries = []  # y-index where a new cluster starts (for divider lines)
y = 0
for name, color, teams in CLUSTERS:
    cluster_boundaries.append((y, name))
    for team in sorted(teams, key=lambda t: data[t][0], reverse=True):
        ordered_teams.append((team, name, color))
        y += 1

n = len(ordered_teams)
y_positions = {team: i for i, (team, _, _) in enumerate(ordered_teams)}

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 9), dpi=200, facecolor="white")
ax.set_facecolor("white")

for team, cluster_name, color in ordered_teams:
    exposure, confirmed = data[team]
    yi = y_positions[team]
    low, high = exposure * RATIO_LOW, exposure * RATIO_HIGH

    ax.plot([low, high], [yi, yi], color=color, alpha=0.35, linewidth=14,
            solid_capstyle="round", zorder=2)

    mult = cluster_multiplier[cluster_name]
    if confirmed is not None:
        ax.plot(confirmed, yi, marker="D", markersize=8, color=color,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)
    elif mult is not None:
        ax.plot(exposure * mult, yi, marker="o", markersize=6, color=color,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)

for team, value, label in NEW_DEALS:
    yi = y_positions[team]
    cluster_color = next(c for t, cn, c in ordered_teams if t == team)
    ax.plot(value, yi, marker="*", markersize=15, color="#eda100",
            markeredgecolor="#0b0b0b", markeredgewidth=0.6, zorder=5)
    ax.annotate(f"{label} ${value:.0f}M", (value, yi), textcoords="offset points",
                xytext=(10, 6), fontsize=7.5, color="#0b0b0b", zorder=5)

# Dashed divider between clusters
for yi, name in cluster_boundaries[1:]:
    ax.axhline(yi - 0.5, color="#c3c2b7", linewidth=0.9, linestyle="--", zorder=1)

ax.set_yticks(range(n))
ax.set_yticklabels([t for t, _, _ in ordered_teams], fontsize=7.3)
ax.invert_yaxis()
ax.set_ylim(n - 0.5, -0.5)

ax.set_xlabel("Estimated jersey patch value ($ millions / year)", fontsize=11, color="#0b0b0b")
ax.grid(True, axis="x", color="#e1e0d9", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#898781")
ax.tick_params(colors="#52514e")

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
legend_handles = [
    Line2D([0], [0], marker="D", linestyle="None", markersize=8, color="#52514e",
           label="Confirmed real price"),
    Line2D([0], [0], marker="*", linestyle="None", markersize=13, color="#eda100",
           markeredgecolor="#0b0b0b", label="New deal (not yet in exposure data)"),
]
for name, color, teams in CLUSTERS:
    mult = cluster_multiplier[name]
    tag = f"x{mult:.2f} from confirmed deals" if mult is not None else "range only, no confirmed deal"
    legend_handles.append(
        Line2D([0], [0], color=color, alpha=0.6, linewidth=8,
               label=f"{name} ({len(teams)} teams) — {tag}")
    )

legend = ax.legend(handles=legend_handles, loc="lower right", frameon=False, fontsize=8.5,
                    title="K-means cluster", title_fontsize=9.5)
legend.get_title().set_color("#0b0b0b")

fig.subplots_adjust(top=0.84, bottom=0.08, left=0.14, right=0.98)
fig.text(0.14, 0.965, "Estimated Jersey Patch Value by Cluster — All 30 Teams",
          fontsize=18, fontweight="bold", color="#0b0b0b", va="top")
fig.text(0.14, 0.915,
          "Bar = exposure x observed ratio bounds (0.196x–0.64x). Dots = exposure x cluster "
          "multiplier, only where real confirmed deals anchor it.\n"
          "Contenders and Off the Radar have no confirmed deal yet, so they show a range only "
          "rather than an invented point estimate.",
          fontsize=10, color="#52514e", va="top", linespacing=1.5)

fig.savefig("cluster_valuation_chart.png", facecolor="white")
plt.close(fig)
print("\nSaved cluster_valuation_chart.png")
