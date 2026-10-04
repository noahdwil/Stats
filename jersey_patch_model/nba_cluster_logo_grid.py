"""
NBA jersey patch valuation - cluster summary graphic.

Takes the same k=4 K-means clustering as nba_kmeans_clustering.py and lays
it out as one labeled column per cluster, with each team's logo inside its
column, so the clusters read as a simple reference chart rather than a
scatter plot.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyBboxPatch
from sklearn.cluster import KMeans

DPI = 200
LOGO_DIR = "logos"
LOGO_FILES = {
    "Oklahoma City Thunder": "nba-oklahoma-city-thunder-logo-480x480.png",
    "San Antonio Spurs": "nba-san-antonio-spurs-logo-480x480.png",
    "Detroit Pistons": "nba-detroit-pistons-logo-480x480.png",
    "Boston Celtics": "nba-boston-celtics-logo-480x480.png",
    "Denver Nuggets": "nba-denver-nuggets-logo-2018-480x480.png",
    "LA Lakers": "nba-los-angeles-lakers-logo-480x480.png",
    "New York Knicks": "nba-new-york-knicks-logo-480x480.png",
    "Cleveland Cavaliers": "Clevelan-Cavaliers-logo-2022-480x480.png",
    "Houston Rockets": "nba-houston-rockets-logo-2020-300x300.png",
    "Minnesota Timberwolves": "nba-minnesota-timberwolves-logo-480x480.png",
    "Atlanta Hawks": "nba-atlanta-hawks-logo-480x480.png",
    "Toronto Raptors": "nba-toronto-raptors-logo-2020-480x480.png",
    "Philadelphia 76ers": "nba-philadelphia-76ers-logo-480x480.png",
    "Orlando Magic": "Orlando-Magic-logo-2025-480x480.png",
    "Phoenix Suns": "nba-phoenix-suns-logo-480x480.png",
    "Charlotte Hornets": "nba-charlotte-hornets-logo-480x480.png",
    "Miami Heat": "nba-miami-heat-logo-480x480.png",
    "Portland Trail Blazers": "nba-portland-trail-blazers-logo-480x480.png",
    "LA Clippers": "NBA-LA-Clippers-logo-2024-480x480.png",
    "Golden State Warriors": "nba-golden-state-warriors-logo-2020-480x480.png",
    "Milwaukee Bucks": "nba-milwaukee-bucks-logo-480x480.png",
    "Chicago Bulls": "nba-chicago-bulls-logo-480x480.png",
    "New Orleans Pelicans": "nba-new-orleans-pelicans-logo-480x480.png",
    "Dallas Mavericks": "nba-dallas-mavericks-logo-480x480.png",
    "Memphis Grizzlies": "nba-memphis-grizzlies-logo-480x480.png",
    "Sacramento Kings": "nba-sacramento-kings-logo-480x480.png",
    "Utah Jazz": "utah-jazz-logo-2022-480x480.png",
    "Brooklyn Nets": "nba-brooklyn-nets-logo-480x480.png",
    "Indiana Pacers": "nba-indiana-pacers-logo-480x480.png",
    "Washington Wizards": "nba-washington-wizards-logo-480x480.png",
}


def px_to_pt(px):
    return px * 72 / DPI


# ---------------------------------------------------------------------------
# Data + clustering (same as nba_kmeans_clustering.py)
# ---------------------------------------------------------------------------
data = {
    "Oklahoma City Thunder":    (0.780, 2, 1080, 14.8, 34),
    "San Antonio Spurs":        (0.756, 2, 1950, 15.5, 22),
    "Detroit Pistons":          (0.732, 2, 659,  4.6,  16),
    "Boston Celtics":           (0.683, 1, 1390, 21.3, 25),
    "Denver Nuggets":           (0.659, 2, 959,  6.8,  26),
    "LA Lakers":                (0.646, 2, 2640, 58.8, 34),
    "New York Knicks":          (0.646, 2, 816,  12.1, 34),
    "Cleveland Cavaliers":      (0.634, 1, 1120, 27.5, 24),
    "Houston Rockets":          (0.634, 2, 1300, 24.5, 28),
    "Minnesota Timberwolves":   (0.598, 1, 1250, 7.0,  28),
    "Atlanta Hawks":            (0.561, 1, 509,  5.7,  13),
    "Toronto Raptors":          (0.561, 1, 459,  9.1,  2),
    "Philadelphia 76ers":       (0.549, 1, 542,  8.6,  14),
    "Orlando Magic":            (0.549, 0, 548,  6.3,  14),
    "Phoenix Suns":             (0.549, 1, 864,  7.6,  9),
    "Charlotte Hornets":        (0.537, 0, 430,  5.6,  3),
    "Miami Heat":               (0.524, 0, 873,  25.7, 5),
    "Portland Trail Blazers":   (0.512, 1, 550,  6.8,  8),
    "LA Clippers":              (0.512, 1, 838,  11.5, 21),
    "Golden State Warriors":    (0.451, 1, 2630, 54.0, 34),
    "Milwaukee Bucks":          (0.390, 1, 614,  10.2, 18),
    "Chicago Bulls":            (0.378, 0, 1060, 30.8, 3),
    "New Orleans Pelicans":     (0.317, 0, 414,  5.9,  2),
    "Dallas Mavericks":         (0.317, 1, 963,  11.8, 23),
    "Memphis Grizzlies":        (0.305, 0, 198,  5.6,  10),
    "Sacramento Kings":         (0.268, 0, 293,  10.7, 9),
    "Utah Jazz":                (0.268, 0, 210,  6.0,  2),
    "Brooklyn Nets":            (0.244, 0, 398,  9.9,  2),
    "Indiana Pacers":           (0.232, 1, 651,  7.8,  9),
    "Washington Wizards":       (0.207, 0, 208,  7.5,  2),
}

raw_cols = ["win_pct", "all_star_count", "impressions_millions", "followers_millions", "national_tv_games"]
df = pd.DataFrame.from_dict(data, orient="index", columns=raw_cols)
df.index.name = "Team"

for col in raw_cols:
    df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

df["team_strength"] = df["z_win_pct"] + df["z_all_star_count"]
df["exposure"] = df["z_impressions_millions"] + df["z_followers_millions"] + df["z_national_tv_games"]

X = df[["team_strength", "exposure"]].values
CHOSEN_K = 4
kmeans = KMeans(n_clusters=CHOSEN_K, n_init=20, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

cluster_means = df.groupby("cluster")[["win_pct", "team_strength", "exposure"]].mean()
print("Cluster averages (win%, team_strength index, exposure index):")
print(cluster_means.round(2))

# Titles derived directly from each cluster's average win% / team_strength /
# exposure (printed above) -- not an arbitrary label.
CLUSTER_TITLES = {
    0: ("Middle of the Pack", "Average record, modest exposure"),
    1: ("Global Brand Giants", "Elite exposure, regardless of record"),
    2: ("Rebuilding, Low Visibility", "Weakest records, least exposure"),
    3: ("Contenders", "Best records, strong exposure"),
}
cluster_colors = plt.get_cmap("tab10", CHOSEN_K)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
LOGO_PX = 54
LOGOS_PER_ROW = 3
ROW_HEIGHT = 0.125
COL_MARGIN = 0.07

fig, ax = plt.subplots(figsize=(16, 11), dpi=DPI, facecolor="white")
ax.set_xlim(0, CHOSEN_K)
ax.set_ylim(0, 1)
ax.axis("off")

fig.subplots_adjust(top=0.82, bottom=0.03, left=0.02, right=0.98)

max_rows = max(int(np.ceil(len(df[df["cluster"] == c]) / LOGOS_PER_ROW)) for c in range(CHOSEN_K))
# Bottom padding must clear the last row's logo AND its text label below it
# (label sits px_to_pt(LOGO_PX)/144 + 0.025 below the logo's own center).
BOTTOM_PAD = px_to_pt(LOGO_PX) / 144 + 0.025 + 0.03
panel_top, panel_bottom = 0.90, 0.90 - (max_rows * ROW_HEIGHT) - BOTTOM_PAD
if panel_bottom < 0.03:
    raise ValueError(f"panel_bottom={panel_bottom:.3f} would clip content below the figure -- "
                      f"reduce ROW_HEIGHT or increase LOGOS_PER_ROW (max_rows={max_rows})")

for cluster_id in range(CHOSEN_K):
    color = cluster_colors(cluster_id)
    x0, x1 = cluster_id + COL_MARGIN, cluster_id + 1 - COL_MARGIN

    panel = FancyBboxPatch((x0, panel_bottom), x1 - x0, panel_top - panel_bottom,
                            boxstyle="round,pad=0,rounding_size=0.02",
                            linewidth=1.4, edgecolor=color, facecolor=color, alpha=0.10,
                            transform=ax.transData, zorder=1)
    ax.add_patch(panel)
    # re-stroke the border at full opacity (the fill alpha above washes out the edge too)
    ax.add_patch(FancyBboxPatch((x0, panel_bottom), x1 - x0, panel_top - panel_bottom,
                                 boxstyle="round,pad=0,rounding_size=0.02",
                                 linewidth=1.4, edgecolor=color, facecolor="none",
                                 zorder=2))

    title, subtitle = CLUSTER_TITLES[cluster_id]
    members = df[df["cluster"] == cluster_id].sort_values("exposure", ascending=False)
    cx = (x0 + x1) / 2

    ax.text(cx, 0.965, title, ha="center", va="top", fontsize=15.5, fontweight="bold",
            color=color, transform=ax.transData, wrap=True)
    ax.text(cx, 0.925, subtitle, ha="center", va="top", fontsize=10, color="#52514e",
            transform=ax.transData)
    ax.text(cx, 0.895, f"{len(members)} teams", ha="center", va="top", fontsize=9,
            color="#898781", style="italic", transform=ax.transData)

    col_span = x1 - x0
    slot_width = col_span / LOGOS_PER_ROW
    for idx, (team, row) in enumerate(members.iterrows()):
        r, c = divmod(idx, LOGOS_PER_ROW)
        lx = x0 + slot_width * (c + 0.5)
        ly = panel_top - 0.10 - r * ROW_HEIGHT

        logo_filename = LOGO_FILES.get(team)
        logo_path = os.path.join(LOGO_DIR, logo_filename) if logo_filename else None
        if logo_path and os.path.exists(logo_path):
            img = plt.imread(logo_path)
            zoom = px_to_pt(LOGO_PX) / img.shape[1]
            imagebox = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(imagebox, (lx, ly), frameon=False, pad=0, zorder=3)
            ax.add_artist(ab)
        else:
            ax.scatter([lx], [ly], s=px_to_pt(LOGO_PX) ** 2, color=color, zorder=3)
            print(f"No logo found for {team}, drew a colored dot instead.")

        ax.text(lx, ly - px_to_pt(LOGO_PX) / 144 - 0.025, team, ha="center", va="top",
                fontsize=8, color="#0b0b0b", transform=ax.transData)

fig.text(0.5, 0.965, "NBA Patch Value — Team Clusters at a Glance",
          fontsize=21, fontweight="bold", color="#0b0b0b", ha="center")
fig.text(0.5, 0.925,
          "K-means clusters (k=4) on Team Strength (win% + All-Star count) and Exposure "
          "(impressions + followers + national TV games), grouped by their shared theme.",
          fontsize=11, color="#52514e", ha="center")

fig.savefig("kmeans_cluster_grid.png", facecolor="white")
plt.close(fig)
print("\nSaved kmeans_cluster_grid.png")
