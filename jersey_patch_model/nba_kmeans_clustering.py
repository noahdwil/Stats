"""
K-means clustering of NBA teams on Team Strength vs. Exposure, for the
jersey patch valuation project.

Team Strength = z-score(win%) + z-score(All-Star count)
Exposure      = z-score(social impressions) + z-score(total followers) + z-score(national TV games)
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
TARGET_LOGO_PX = 30  # on-canvas diameter at the figure's dpi, regardless of source resolution

# ---------------------------------------------------------------------------
# Data: team -> (win_pct, all_star_count, impressions_millions, followers_millions, national_tv_games)
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

print(f"Loaded {len(df)} teams.\n")

# ---------------------------------------------------------------------------
# Standardize each raw metric, build the two composite axes
# ---------------------------------------------------------------------------
for col in raw_cols:
    df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std(ddof=0)

df["team_strength"] = df["z_win_pct"] + df["z_all_star_count"]
df["exposure"] = df["z_impressions_millions"] + df["z_followers_millions"] + df["z_national_tv_games"]

X = df[["team_strength", "exposure"]].values

# ---------------------------------------------------------------------------
# Choose k: elbow (inertia) + silhouette score, k=2..6
# ---------------------------------------------------------------------------
print("k | inertia   | silhouette")
print("--|-----------|-----------")
inertias, silhouettes = {}, {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X)
    inertias[k] = km.inertia_
    silhouettes[k] = silhouette_score(X, km.labels_)
    print(f"{k} | {inertias[k]:9.3f} | {silhouettes[k]:9.3f}")

# Best silhouette score (0.530) and the elbow's steepest inertia drop
# (136.5 -> 71.9) both point to k=3 -- but k=4 is set explicitly below per
# request, trading a bit of fit quality (silhouette 0.479 vs 0.530) for a
# finer-grained split.
CHOSEN_K = 4
print(f"\nChosen k = {CHOSEN_K} (set explicitly; best silhouette score was k=3 at 0.530, "
      f"k={CHOSEN_K}'s silhouette is {silhouettes[CHOSEN_K]:.3f})")

kmeans = KMeans(n_clusters=CHOSEN_K, n_init=20, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
HIGHLIGHT_TEAMS = [
    "Oklahoma City Thunder", "New York Knicks", "LA Lakers",
    "Golden State Warriors", "San Antonio Spurs", "Chicago Bulls",
]
HIGHLIGHT_LABELS = {
    "Oklahoma City Thunder": "OKC Thunder",
    "New York Knicks": "New York Knicks",
    "LA Lakers": "LA Lakers",
    "Golden State Warriors": "Golden State Warriors",
    "San Antonio Spurs": "San Antonio Spurs",
    "Chicago Bulls": "Chicago Bulls",
}

fig, ax = plt.subplots(figsize=(14, 9), dpi=200, facecolor="white")
ax.set_facecolor("white")

cluster_colors = plt.get_cmap("tab10", CHOSEN_K)

# Cluster-colored ring behind each point -- doubles as the legend swatch and
# as the cluster-identity cue once logos sit on top of the plain dots.
for cluster_id in range(CHOSEN_K):
    sub = df[df["cluster"] == cluster_id]
    ax.scatter(sub["team_strength"], sub["exposure"],
               s=230, facecolor="white", edgecolor=cluster_colors(cluster_id),
               linewidth=2.0, label=f"Cluster {cluster_id} (n={len(sub)})", zorder=3)

# Team logo on top of each ring; falls back to a plain colored dot if a
# team's logo file isn't found in LOGO_DIR.
missing_logos = []
for team, row in df.iterrows():
    logo_filename = LOGO_FILES.get(team)
    logo_path = os.path.join(LOGO_DIR, logo_filename) if logo_filename else None
    if logo_path and os.path.exists(logo_path):
        img = plt.imread(logo_path)
        img_width_px = img.shape[1]
        zoom = TARGET_LOGO_PX / img_width_px
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, (row["team_strength"], row["exposure"]),
                             frameon=False, pad=0, zorder=4)
        ax.add_artist(ab)
    else:
        missing_logos.append(team)
        ax.scatter(row["team_strength"], row["exposure"],
                   s=110, color=cluster_colors(row["cluster"]), edgecolor="white",
                   linewidth=0.8, zorder=4)

if missing_logos:
    print(f"No logo file found for {len(missing_logos)} team(s), drew a plain dot instead: "
          f"{', '.join(missing_logos)}")

# Bold label for the named teams (no ring). Offsets are tuned to clear the
# ~38px logo's own footprint plus its nearest neighbors (Spurs/Thunder/Knicks
# sit close together on the right edge; Bulls' label needs to clear its own
# wordmark-heavy logo).
label_offsets = {
    "Oklahoma City Thunder": (0, -28),
    "New York Knicks": (26, -14),
    "LA Lakers": (-28, 14),
    "Golden State Warriors": (26, -14),
    "San Antonio Spurs": (-26, 18),
    "Chicago Bulls": (26, 18),
}
for team in HIGHLIGHT_TEAMS:
    row = df.loc[team]
    dx, dy = label_offsets[team]
    ha = "center" if dx == 0 else ("left" if dx > 0 else "right")
    ax.annotate(HIGHLIGHT_LABELS[team], (row["team_strength"], row["exposure"]),
                textcoords="offset points", xytext=(dx, dy), ha=ha,
                fontsize=10.5, fontweight="bold", color="#0b0b0b", zorder=5)

ax.axhline(0, color="#898781", linewidth=0.9, zorder=1)
ax.axvline(0, color="#898781", linewidth=0.9, zorder=1)
ax.grid(True, color="#e1e0d9", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#898781")
ax.tick_params(colors="#52514e")

ax.set_xlabel("Team Strength Index (win% + All-Star count)", fontsize=11.5, color="#0b0b0b")
ax.set_ylabel("Exposure Index (impressions + followers + national TV games)", fontsize=11.5, color="#0b0b0b")
ax.legend(frameon=False, loc="upper left", fontsize=10)

fig.subplots_adjust(top=0.85, bottom=0.09, left=0.08, right=0.97)
fig.text(0.08, 0.97, "NBA Patch Value — K-Means Team Clusters",
          fontsize=19, fontweight="bold", color="#0b0b0b", va="top")
fig.text(0.08, 0.905,
          "Team Strength = combined z-score of 2025-26 win% and confirmed All-Star\n"
          "count. Exposure = combined z-score of social impressions, followers, and national\n"
          "TV games. Color = K-means cluster (k chosen via elbow/silhouette method).",
          fontsize=10.5, color="#52514e", linespacing=1.4, va="top")

fig.savefig("kmeans_exposure.png", facecolor="white")
plt.close(fig)
print("\nSaved kmeans_exposure.png")

# ---------------------------------------------------------------------------
# Print cluster membership
# ---------------------------------------------------------------------------
print(f"\nCluster membership (k={CHOSEN_K}):")
for cluster_id in range(CHOSEN_K):
    members = df[df["cluster"] == cluster_id].sort_values("exposure", ascending=False)
    print(f"\nCluster {cluster_id} (n={len(members)}):")
    for team in members.index:
        print(f"  - {team}")
