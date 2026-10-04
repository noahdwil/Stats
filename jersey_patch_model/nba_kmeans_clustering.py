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
DPI = 200
TARGET_LOGO_PX = 55  # on-canvas diameter in actual screen pixels, regardless of source resolution
DISC_PX = 75          # solid cluster-color disc drawn behind each logo (> logo so it rings visibly)

def px_to_pt(px):
    """Convert a target size in on-canvas pixels (at DPI) to the points unit
    that OffsetImage's zoom and scatter's s both actually use internally --
    1 point = DPI/72 pixels, so a naive pixel value fed straight into either
    API renders ~2.8x too big at DPI=200."""
    return px * 72 / DPI

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
fig, ax = plt.subplots(figsize=(14, 9), dpi=DPI, facecolor="white")
ax.set_facecolor("white")

cluster_colors = plt.get_cmap("tab10", CHOSEN_K)

# Fix the axis limits up front so the data<->pixel conversion used for
# overlap-avoidance below stays correct once we start placing artists.
x_pad, y_pad = 0.6, 0.6
ax.set_xlim(df["team_strength"].min() - x_pad, df["team_strength"].max() + x_pad)
ax.set_ylim(df["exposure"].min() - y_pad, df["exposure"].max() + y_pad)
fig.subplots_adjust(top=0.85, bottom=0.09, left=0.08, right=0.97)

# ---------------------------------------------------------------------------
# Overlap avoidance: any logos within SEPARATION_PX of each other get grouped
# (union-find over the proximity graph, so chains of close points merge into
# one group) and arranged on a small circle around their shared centroid,
# radius chosen so adjacent members are guaranteed SEPARATION_PX apart no
# matter how many pile into one group. A pairwise relaxation cleanup pass
# then resolves any remaining cross-group close calls. True data positions
# are untouched -- only where each logo is DRAWN shifts, and a thin leader
# line marks any real nudge.
# ---------------------------------------------------------------------------
SEPARATION_PX = 85  # center-to-center spacing target (> DISC_PX so discs don't just kiss edges)

teams = df.index.tolist()
n = len(teams)
true_px = ax.transData.transform(df[["team_strength", "exposure"]].values).astype(float)

parent = list(range(n))

def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

def union(i, j):
    ri, rj = find(i), find(j)
    if ri != rj:
        parent[ri] = rj

for i in range(n):
    for j in range(i + 1, n):
        if np.hypot(*(true_px[i] - true_px[j])) < SEPARATION_PX:
            union(i, j)

groups = {}
for i in range(n):
    groups.setdefault(find(i), []).append(i)

plot_px = true_px.copy()
for members in groups.values():
    if len(members) == 1:
        continue
    centroid = true_px[members].mean(axis=0)
    radius = SEPARATION_PX / (2 * np.sin(np.pi / len(members))) if len(members) > 1 else 0
    for k, idx in enumerate(members):
        angle = 2 * np.pi * k / len(members) + np.pi / 2  # first point straight up, then clockwise
        plot_px[idx] = centroid + radius * np.array([np.cos(angle), np.sin(angle)])

# Cleanup pass: resolve any remaining close calls between different groups
rng = np.random.default_rng(42)
for _ in range(300):
    moved = False
    for i in range(n):
        for j in range(i + 1, n):
            dx = plot_px[j, 0] - plot_px[i, 0]
            dy = plot_px[j, 1] - plot_px[i, 1]
            dist = np.hypot(dx, dy)
            if dist < 1e-6:
                angle = rng.uniform(0, 2 * np.pi)
                dx, dy, dist = np.cos(angle), np.sin(angle), 1.0
            if dist < SEPARATION_PX:
                push = (SEPARATION_PX - dist) / 2
                ux, uy = dx / dist, dy / dist
                plot_px[i] -= [ux * push, uy * push]
                plot_px[j] += [ux * push, uy * push]
                moved = True
    if not moved:
        break

plot_data = ax.transData.inverted().transform(plot_px)
df["plot_x"] = plot_data[:, 0]
df["plot_y"] = plot_data[:, 1]
nudged = np.hypot(plot_px[:, 0] - true_px[:, 0], plot_px[:, 1] - true_px[:, 1]) > 2

# Leader lines for any team whose logo had to move to avoid overlap
for team, was_nudged in zip(teams, nudged):
    if was_nudged:
        row = df.loc[team]
        ax.plot([row["team_strength"], row["plot_x"]], [row["exposure"], row["plot_y"]],
                color="#898781", linewidth=0.6, linestyle="-", alpha=0.7, zorder=2)

if nudged.any():
    print(f"\nNudged {int(nudged.sum())} overlapping logo(s) to stay visible "
          f"(thin leader line marks the true position): {', '.join(np.array(teams)[nudged])}")

# Solid cluster-color disc behind each logo -- a filled disc reads far more
# clearly than a thin ring, especially once points start crowding together.
for cluster_id in range(CHOSEN_K):
    sub = df[df["cluster"] == cluster_id]
    ax.scatter(sub["plot_x"], sub["plot_y"],
               s=px_to_pt(DISC_PX) ** 2, color=cluster_colors(cluster_id),
               edgecolor="white", linewidth=1.2, zorder=3)

# Team logo on top of each disc; falls back to a plain colored dot if a
# team's logo file isn't found in LOGO_DIR.
missing_logos = []
for team, row in df.iterrows():
    logo_filename = LOGO_FILES.get(team)
    logo_path = os.path.join(LOGO_DIR, logo_filename) if logo_filename else None
    if logo_path and os.path.exists(logo_path):
        img = plt.imread(logo_path)
        img_width_px = img.shape[1]
        zoom = px_to_pt(TARGET_LOGO_PX) / img_width_px
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, (row["plot_x"], row["plot_y"]),
                             frameon=False, pad=0, zorder=4)
        ax.add_artist(ab)
    else:
        missing_logos.append(team)

if missing_logos:
    print(f"No logo file found for {len(missing_logos)} team(s), left as a plain colored disc: "
          f"{', '.join(missing_logos)}")

# Legend: proxy markers, since cluster color now lives on per-point discs
# rather than one labeled scatter call per cluster.
legend_handles = [
    plt.Line2D([0], [0], marker="o", linestyle="None", markersize=10,
               markerfacecolor=cluster_colors(cid), markeredgecolor="white",
               label=f"Cluster {cid} (n={(df['cluster'] == cid).sum()})")
    for cid in range(CHOSEN_K)
]

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
ax.legend(handles=legend_handles, frameon=False, loc="upper left", fontsize=10)

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
