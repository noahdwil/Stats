"""
NBA jersey patch valuation - K-means clustering v2: metro population swapped
in for total followers in the Exposure composite.

Team Strength = z(win_pct) + z(all_star_count)
Exposure      = z(impressions_millions) + z(metro_pop_millions) + z(national_tv_games)
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

LOGO_DIR = "logos"
LOGO_FILES = {
    "OKC Thunder": "nba-oklahoma-city-thunder-logo-480x480.png",
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

pd.set_option("display.width", 140)


def section(title):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


# ---------------------------------------------------------------------------
# Data: team -> (win_pct, all_star_count, impressions_millions, national_tv_games, metro_pop_millions)
#
# metro_pop_millions sourced from the attached us-metro-areas-v2025 CSV
# (total_pop_2025, CBSA-level), matched by each team's home market:
#   Knicks & Nets       -> New York-Newark-Jersey City CBSA (shared, as specified)
#   Lakers & Clippers    -> Los Angeles-Long Beach-Anaheim CBSA (shared, as specified)
#   Warriors             -> San Francisco-Oakland-Fremont CBSA
#   Mavericks             -> Dallas-Fort Worth-Arlington CBSA
#   Wizards               -> Washington-Arlington-Alexandria CBSA
# Toronto Raptors: not in the attached US-only CBSA CSV (confirmed absent).
# User-specified source: macrotrends.net/global-metrics/cities/20402/toronto/population
# -> 6,491,000 (2025 estimate). Note this uses a different metro-area
# definition than the US Census CBSA figures used for the other 29 teams
# (Macrotrends' 2025 Toronto figure is ~6.49M; Statistics Canada's official
# CMA figure is ~7.1M for the same year) -- flagged, not silently reconciled.
# ---------------------------------------------------------------------------
data = {
    "OKC Thunder":            (0.780, 2, 1080, 34, 1.512813),
    "San Antonio Spurs":      (0.756, 2, 1950, 22, 2.813140),
    "Detroit Pistons":        (0.732, 2, 659,  16, 4.390913),
    "Boston Celtics":         (0.683, 1, 1390, 25, 5.034221),
    "Denver Nuggets":         (0.659, 2, 959,  26, 3.092037),
    "LA Lakers":              (0.646, 2, 2640, 34, 12.844441),
    "New York Knicks":        (0.646, 2, 816,  34, 20.112448),
    "Cleveland Cavaliers":    (0.634, 1, 1120, 24, 2.165775),
    "Houston Rockets":        (0.634, 2, 1300, 28, 7.904627),
    "Minnesota Timberwolves": (0.598, 1, 1250, 28, 3.790295),
    "Atlanta Hawks":          (0.561, 1, 509,  13, 6.482182),
    "Toronto Raptors":        (0.561, 1, 459,  2,  6.491),
    "Philadelphia 76ers":     (0.549, 1, 542,  14, 6.329118),
    "Orlando Magic":          (0.549, 0, 548,  14, 2.957672),
    "Phoenix Suns":           (0.549, 1, 864,  9,  5.228938),
    "Charlotte Hornets":      (0.537, 0, 430,  3,  2.938830),
    "Miami Heat":             (0.524, 0, 873,  5,  6.391072),
    "Portland Trail Blazers": (0.512, 1, 550,  8,  2.542282),
    "LA Clippers":            (0.512, 1, 838,  21, 12.844441),
    "Golden State Warriors":  (0.451, 1, 2630, 34, 4.630041),
    "Milwaukee Bucks":        (0.390, 1, 614,  18, 1.575010),
    "Chicago Bulls":          (0.378, 0, 1060, 3,  9.434123),
    "New Orleans Pelicans":   (0.317, 0, 414,  2,  0.970849),
    "Dallas Mavericks":       (0.317, 1, 963,  23, 8.477157),
    "Memphis Grizzlies":      (0.305, 0, 198,  10, 1.341412),
    "Sacramento Kings":       (0.268, 0, 293,  9,  2.477274),
    "Utah Jazz":              (0.268, 0, 210,  2,  1.308377),
    "Brooklyn Nets":          (0.244, 0, 398,  2,  20.112448),
    "Indiana Pacers":         (0.232, 1, 651,  9,  2.205695),
    "Washington Wizards":     (0.207, 0, 208,  2,  6.465724),
}

PREVIOUS_CLUSTERS = {
    "Household Names": ["LA Lakers", "Golden State Warriors"],
    "Contenders": ["Houston Rockets", "San Antonio Spurs", "Boston Celtics", "Cleveland Cavaliers",
                   "OKC Thunder", "New York Knicks", "Minnesota Timberwolves", "Denver Nuggets", "Detroit Pistons"],
    "Middle of the Pack": ["Dallas Mavericks", "Chicago Bulls", "LA Clippers", "Miami Heat",
                            "Milwaukee Bucks", "Philadelphia 76ers", "Phoenix Suns", "Orlando Magic",
                            "Atlanta Hawks", "Portland Trail Blazers", "Toronto Raptors"],
    "Off the Radar": ["Indiana Pacers", "Sacramento Kings", "Memphis Grizzlies", "Brooklyn Nets",
                       "Charlotte Hornets", "New Orleans Pelicans", "Washington Wizards", "Utah Jazz"],
}
PREVIOUS_SILHOUETTE = {4: 0.479, 3: 0.530}

# Old model's raw followers_millions (from nba_kmeans_clustering.py), needed
# for the "before" correlation comparison in Step 5.
OLD_FOLLOWERS_MILLIONS = {
    "Oklahoma City Thunder": 14.8, "San Antonio Spurs": 15.5, "Detroit Pistons": 4.6,
    "Boston Celtics": 21.3, "Denver Nuggets": 6.8, "LA Lakers": 58.8, "New York Knicks": 12.1,
    "Cleveland Cavaliers": 27.5, "Houston Rockets": 24.5, "Minnesota Timberwolves": 7.0,
    "Atlanta Hawks": 5.7, "Toronto Raptors": 9.1, "Philadelphia 76ers": 8.6, "Orlando Magic": 6.3,
    "Phoenix Suns": 7.6, "Charlotte Hornets": 5.6, "Miami Heat": 25.7, "Portland Trail Blazers": 6.8,
    "LA Clippers": 11.5, "Golden State Warriors": 54.0, "Milwaukee Bucks": 10.2, "Chicago Bulls": 30.8,
    "New Orleans Pelicans": 5.9, "Dallas Mavericks": 11.8, "Memphis Grizzlies": 5.6,
    "Sacramento Kings": 10.7, "Utah Jazz": 6.0, "Brooklyn Nets": 9.9, "Indiana Pacers": 7.8,
    "Washington Wizards": 7.5,
}
NAME_FIX = {"Oklahoma City Thunder": "OKC Thunder"}  # naming diff between old/new scripts
OLD_FOLLOWERS_MILLIONS = {NAME_FIX.get(k, k): v for k, v in OLD_FOLLOWERS_MILLIONS.items()}


# ---------------------------------------------------------------------------
# Step 8 (done first on purpose): validation
# ---------------------------------------------------------------------------
section("VALIDATION")

missing = [team for team, vals in data.items() if vals[4] is None]
if missing:
    raise ValueError(
        f"metro_pop_millions is None for: {', '.join(missing)}. "
        f"The attached metro-areas CSV is US-only (CBSA) and has no entry for these teams' "
        f"markets -- nothing was invented to fill the gap. Supply the missing value(s) "
        f"(e.g. a cited Statistics Canada figure for Toronto) and re-run."
    )

raw_cols = ["win_pct", "all_star_count", "impressions_millions", "national_tv_games", "metro_pop_millions"]
df = pd.DataFrame.from_dict(data, orient="index", columns=raw_cols)
df.index.name = "Team"

warnings = []
if df["win_pct"].max() > 1 or df["win_pct"].min() < 0:
    warnings.append("win_pct outside [0,1]")
if (df["all_star_count"] < 0).any() or (df["all_star_count"] > 4).any():
    warnings.append("all_star_count outside a plausible 0-4 range")
if (df["metro_pop_millions"] <= 0).any():
    warnings.append("metro_pop_millions <= 0 for at least one team")
dup_pop_not_shared = df.groupby("metro_pop_millions").size()
expected_shared = {12.844441: 2, 20.112448: 2}
for pop_val, count in dup_pop_not_shared.items():
    if count > 1 and round(pop_val, 6) not in expected_shared:
        warnings.append(f"Unexpected shared metro_pop_millions value {pop_val} across {count} teams")

if warnings:
    print("WARNINGS:")
    for w in warnings:
        print(f"  - {w}")
else:
    print("No inconsistencies detected in the input data.")


# ---------------------------------------------------------------------------
# Step 1: composite indices
# ---------------------------------------------------------------------------
section("STEP 1 - Team Strength and Exposure indices")

strength_scaler = StandardScaler()
exposure_scaler = StandardScaler()
z_strength = strength_scaler.fit_transform(df[["win_pct", "all_star_count"]])
z_exposure = exposure_scaler.fit_transform(df[["impressions_millions", "metro_pop_millions", "national_tv_games"]])

df["team_strength"] = z_strength.sum(axis=1)
df["exposure"] = z_exposure.sum(axis=1)
df["z_impressions"] = z_exposure[:, 0]
df["z_metro_pop"] = z_exposure[:, 1]
df["z_national_tv_games"] = z_exposure[:, 2]

print(df[["team_strength", "exposure"]].round(3).to_string())


# ---------------------------------------------------------------------------
# Step 2: choose k
# ---------------------------------------------------------------------------
section("STEP 2 - Choosing k (elbow + silhouette)")

X = df[["team_strength", "exposure"]].values
print("k | inertia   | silhouette")
print("--|-----------|-----------")
inertias, silhouettes = {}, {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(X)
    inertias[k] = km.inertia_
    silhouettes[k] = silhouette_score(X, km.labels_)
    print(f"{k} | {inertias[k]:9.3f} | {silhouettes[k]:9.3f}")

best_sil_k = max(silhouettes, key=silhouettes.get)
inertia_drops = {k: inertias[k - 1] - inertias[k] for k in range(3, 7)}
elbow_k = max(inertia_drops, key=inertia_drops.get)

print(f"\nBest silhouette: k={best_sil_k} ({silhouettes[best_sil_k]:.3f}). "
      f"Steepest inertia drop: k={elbow_k} (inertia fell by {inertia_drops[elbow_k]:.1f} going into it).")

if best_sil_k == elbow_k:
    print(f"Elbow and silhouette agree on k={best_sil_k}.")
else:
    print(f"Elbow and silhouette DISAGREE (elbow points to k={elbow_k}, silhouette to k={best_sil_k}).")

CHOSEN_K = 4
print(f"Chosen k = {CHOSEN_K} (set explicitly, for continuity with the previous 4-cluster model). "
      f"Neither metric actually picks k=4 on its own here: silhouette peaks at k={best_sil_k} "
      f"({silhouettes[best_sil_k]:.3f} vs. k=4's {silhouettes[4]:.3f}), and the steepest inertia "
      f"drop is at k={elbow_k}, not 4 -- noted plainly rather than overstating the fit.")

kmeans = KMeans(n_clusters=CHOSEN_K, n_init=20, random_state=42)
df["cluster"] = kmeans.fit_predict(X)


# ---------------------------------------------------------------------------
# Step 3: cluster summaries + suggested names
# ---------------------------------------------------------------------------
section("STEP 3 - Cluster summaries")

cluster_summary = df.groupby("cluster")[["team_strength", "exposure"]].mean()
print("Centroids:")
print(cluster_summary.round(3).to_string())

# Rank-based naming so names stay distinct regardless of k: each cluster's
# label comes from its RANK on each axis (relative to the other clusters),
# not a fixed absolute threshold -- fixed thresholds collided for k=4 here
# (two low-exposure clusters both landed under the same "Low-Exposure" cutoff).
STRENGTH_VOCAB = ["Strongest", "Stronger", "Strong", "Average", "Weak", "Weaker", "Weakest"]
EXPOSURE_VOCAB = ["Highest-Exposure", "Higher-Exposure", "High-Exposure", "Mid-Exposure",
                  "Low-Exposure", "Lower-Exposure", "Lowest-Exposure"]

def rank_labels(k, vocab):
    if k == 1:
        return [vocab[len(vocab) // 2]]
    idx = np.round(np.linspace(0, len(vocab) - 1, k)).astype(int)
    return [vocab[i] for i in idx]

strength_order = cluster_summary["team_strength"].sort_values(ascending=False).index.tolist()
exposure_order = cluster_summary["exposure"].sort_values(ascending=False).index.tolist()
strength_label_for = dict(zip(strength_order, rank_labels(CHOSEN_K, STRENGTH_VOCAB)))
exposure_label_for = dict(zip(exposure_order, rank_labels(CHOSEN_K, EXPOSURE_VOCAB)))

cluster_names = {}
for cid, row in cluster_summary.iterrows():
    name = f"{strength_label_for[cid]}, {exposure_label_for[cid]}"
    cluster_names[cid] = name
    members = sorted(df[df["cluster"] == cid].index.tolist())
    print(f"\nCluster {cid} - suggested name: \"{name}\"  (n={len(members)}, "
          f"centroid strength={row['team_strength']:.2f}, exposure={row['exposure']:.2f})")
    for m in members:
        print(f"  - {m}")

if len(set(cluster_names.values())) < len(cluster_names):
    print("\nWARNING: two clusters still produced the same suggested name -- "
          "check cluster_summary above and name them manually for the poster.")


# ---------------------------------------------------------------------------
# Step 4: compare to previous model
# ---------------------------------------------------------------------------
section("STEP 4 - Comparison to the previous (followers-based) model")

old_assignment = {}
for cname, members in PREVIOUS_CLUSTERS.items():
    for m in members:
        old_assignment[m] = cname

new_assignment = df["cluster"].map(cluster_names).to_dict()

changed = []
for team in df.index:
    old_c = old_assignment.get(team, "(not in old model)")
    new_c = new_assignment[team]
    if old_c != "(not in old model)":
        changed.append((team, old_c, new_c))

print("Team-by-team (old cluster -> new cluster):")
any_changed = False
for team, old_c, new_c in changed:
    marker = ""
    print(f"  {team}: {old_c}")

# Closest-matching old cluster per new cluster (computed once, reused below
# and for the chart legend, which uses these familiar titles per request
# instead of the data-driven rank descriptors from Step 3).
closest_old_cluster = {}
for cid in sorted(df["cluster"].unique()):
    new_members = set(df[df["cluster"] == cid].index)
    best_old, best_overlap = None, -1
    for old_name, old_members in PREVIOUS_CLUSTERS.items():
        overlap = len(new_members & set(old_members))
        if overlap > best_overlap:
            best_overlap, best_old = overlap, old_name
    closest_old_cluster[cid] = best_old
    print(f"\nNew Cluster {cid} (\"{cluster_names[cid]}\", n={len(new_members)}) most resembles "
          f"old cluster \"{best_old}\" ({best_overlap}/{len(new_members)} members overlap)")

print("\nTeams that changed cluster membership (old grouping name vs. nearest-match new grouping):")
moved_any = False
for cid in sorted(df["cluster"].unique()):
    new_members = set(df[df["cluster"] == cid].index)
    best_old = closest_old_cluster[cid]
    moved = new_members - set(PREVIOUS_CLUSTERS[best_old])
    for team in sorted(moved):
        print(f"  {team}: was in \"{old_assignment.get(team)}\", now in Cluster {cid} "
              f"(\"{cluster_names[cid]}\", closest old match \"{best_old}\")")
        moved_any = True
if not moved_any:
    print("  (none -- every new cluster's members are a subset of its closest old cluster)")


# ---------------------------------------------------------------------------
# Step 5: does metro population dominate Exposure?
# ---------------------------------------------------------------------------
section("STEP 5 - Does metro population dominate the Exposure index?")

r_new, p_new = pearsonr(df["exposure"], df["metro_pop_millions"])
rho_new, p_rho_new = spearmanr(df["exposure"], df["metro_pop_millions"])
print(f"NEW Exposure index vs metro_pop_millions: Pearson r={r_new:.3f} (p={p_new:.3f}), "
      f"Spearman rho={rho_new:.3f} (p={p_rho_new:.3f})")
print("CAVEAT: metro_pop_millions is literally one of the three terms summed into this Exposure "
      "index, so part of this correlation is mechanical/tautological, not an independent finding.")

old_exposure_values = []
metro_for_old = []
for team in df.index:
    if team in OLD_FOLLOWERS_MILLIONS:
        old_exposure_values.append(team)
old_df = df.loc[old_exposure_values, ["impressions_millions", "national_tv_games", "metro_pop_millions"]].copy()
old_df["followers_millions"] = [OLD_FOLLOWERS_MILLIONS[t] for t in old_df.index]
old_scaler = StandardScaler()
z_old = old_scaler.fit_transform(old_df[["impressions_millions", "followers_millions", "national_tv_games"]])
old_df["old_exposure"] = z_old.sum(axis=1)

r_old, p_old = pearsonr(old_df["old_exposure"], old_df["metro_pop_millions"])
rho_old, p_rho_old = spearmanr(old_df["old_exposure"], old_df["metro_pop_millions"])
print(f"\nBEFORE (old, followers-based Exposure index) vs metro_pop_millions -- population was NOT "
      f"part of that formula, so this is an honest independent check:")
print(f"  Pearson r={r_old:.3f} (p={p_old:.3f}), Spearman rho={rho_old:.3f} (p={p_rho_old:.3f})")
print(f"  (For reference, the project's earlier DMA-rank-vs-dollar-exposure check found r=+0.24, p=0.21.)")

print(f"\nSummary: old exposure-vs-population relationship was r={r_old:.3f}; now that population is "
      f"INSIDE the formula, the correlation is {'much ' if abs(r_new) - abs(r_old) > 0.3 else ''}"
      f"higher (r={r_new:.3f}) almost by construction.")

# Variance-share check: each z-component has unit variance individually
# (StandardScaler, population std); their contribution to the summed
# Exposure index's variance depends on their pairwise correlations.
var_components = df[["z_impressions", "z_metro_pop", "z_national_tv_games"]].var(ddof=0)
var_exposure = df["exposure"].var(ddof=0)
corr_with_composite = df[["z_impressions", "z_metro_pop", "z_national_tv_games"]].corrwith(df["exposure"])
print(f"\nEach standardized component's own variance (should be ~1.0 by construction):")
print(var_components.round(3).to_string())
print(f"\nVariance of the summed Exposure index: {var_exposure:.3f} (3.0 would mean the three "
      f"components are completely uncorrelated with each other)")
print(f"\nCorrelation of each component with the final Exposure composite (practical 'who dominates' check):")
print(corr_with_composite.round(3).to_string())
dominant = corr_with_composite.idxmax()
print(f"\n{'metro_pop' if 'metro_pop' in dominant else dominant} has the highest correlation with the "
      f"composite -- {'population DOES look like the main driver of Exposure' if dominant == 'z_metro_pop' else 'population is not the single dominant driver of Exposure'}.")


# ---------------------------------------------------------------------------
# Step 6: visualization
# ---------------------------------------------------------------------------
section("STEP 6 - Visualization")

DPI = 200
TARGET_LOGO_PX = 55   # on-canvas logo diameter in actual screen pixels
DISC_PX = 75          # cluster-color disc behind each logo
SEPARATION_PX = 85    # minimum center-to-center spacing enforced between any two logos

def px_to_pt(px):
    """OffsetImage's zoom and scatter's s are both in points (1pt = dpi/72 px
    at this figure's dpi) -- convert so 'pixels' means actual screen pixels."""
    return px * 72 / DPI

cluster_colors = plt.get_cmap("tab10", CHOSEN_K)

fig, ax = plt.subplots(figsize=(16, 9), dpi=DPI, facecolor="white")
ax.set_facecolor("white")

x_pad, y_pad = 0.6, 0.6
ax.set_xlim(df["team_strength"].min() - x_pad, df["team_strength"].max() + x_pad)
ax.set_ylim(df["exposure"].min() - y_pad, df["exposure"].max() + y_pad)
fig.subplots_adjust(top=0.85, bottom=0.08, left=0.06, right=0.97)

# ---------------------------------------------------------------------------
# Overlap avoidance: group any logos within SEPARATION_PX of each other
# (union-find over the proximity graph) and arrange each group on a small
# circle around their shared true centroid, radius chosen so every member is
# guaranteed SEPARATION_PX from its neighbors regardless of group size. A
# pairwise-relaxation cleanup pass then resolves any remaining cross-group
# close calls. True data positions are untouched -- only the drawn position
# shifts, and a thin leader line marks any real nudge.
# ---------------------------------------------------------------------------
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
    radius = SEPARATION_PX / (2 * np.sin(np.pi / len(members)))
    for k, idx in enumerate(members):
        angle = 2 * np.pi * k / len(members) + np.pi / 2
        plot_px[idx] = centroid + radius * np.array([np.cos(angle), np.sin(angle)])

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

for team, was_nudged in zip(teams, nudged):
    if was_nudged:
        row = df.loc[team]
        ax.plot([row["team_strength"], row["plot_x"]], [row["exposure"], row["plot_y"]],
                color="#898781", linewidth=0.6, linestyle="-", alpha=0.7, zorder=2)

if nudged.any():
    print(f"Nudged {int(nudged.sum())} overlapping logo(s) to stay visible "
          f"(thin leader line marks the true position): {', '.join(np.array(teams)[nudged])}")

# Solid cluster-color disc behind each logo, then the logo on top
for cid in range(CHOSEN_K):
    sub = df[df["cluster"] == cid]
    ax.scatter(sub["plot_x"], sub["plot_y"], s=px_to_pt(DISC_PX) ** 2, color=cluster_colors(cid),
               edgecolor="white", linewidth=1.2,
               label=f"{closest_old_cluster[cid]} (n={len(sub)})", zorder=3)

missing_logos = []
for team, row in df.iterrows():
    logo_filename = LOGO_FILES.get(team)
    logo_path = os.path.join(LOGO_DIR, logo_filename) if logo_filename else None
    if logo_path and os.path.exists(logo_path):
        img = plt.imread(logo_path)
        zoom = px_to_pt(TARGET_LOGO_PX) / img.shape[1]
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, (row["plot_x"], row["plot_y"]), frameon=False, pad=0, zorder=4)
        ax.add_artist(ab)
    else:
        missing_logos.append(team)

if missing_logos:
    print(f"No logo file found for {len(missing_logos)} team(s), left as a plain colored disc: "
          f"{', '.join(missing_logos)}")

ax.axhline(0, color="#898781", linewidth=0.9, zorder=1)
ax.axvline(0, color="#898781", linewidth=0.9, zorder=1)
ax.grid(True, color="#e1e0d9", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#898781")
ax.tick_params(colors="#52514e")
ax.set_xlabel("Team Strength Index", fontsize=11.5, color="#0b0b0b")
ax.set_ylabel("Exposure Index", fontsize=11.5, color="#0b0b0b")
ax.legend(frameon=False, loc="upper left", fontsize=9.5)

fig.text(0.06, 0.965, "NBA Patch Value: K-Means Team Clusters", fontsize=20, fontweight="bold",
          color="#0b0b0b", va="top")
fig.text(0.06, 0.915,
          "Team Strength = win% + All-Star count. Exposure = social impressions + metro population + "
          "national TV games. Color = K-means cluster.",
          fontsize=11, color="#52514e", va="top")

fig.savefig("kmeans_metro_pop.png", facecolor="white")
plt.close(fig)
print("Saved kmeans_metro_pop.png")


# ---------------------------------------------------------------------------
# Step 7: save CSV + slide snippet
# ---------------------------------------------------------------------------
section("STEP 7 - Save results + slide snippet")

out_cols = ["win_pct", "all_star_count", "impressions_millions", "national_tv_games",
            "metro_pop_millions", "team_strength", "exposure", "cluster"]
out = df[out_cols].copy()
out["cluster_name"] = out["cluster"].map(cluster_names)
out.to_csv("kmeans_metro_pop_results.csv")
print("Saved kmeans_metro_pop_results.csv")

print("""
Slide snippet:
  # Features: win%, All-Star count, social impressions, metro population, national TV games
  X_scaled = StandardScaler().fit_transform(df[features])
  Team_Strength, Exposure = z(win_pct)+z(all_stars),  z(impressions)+z(metro_pop)+z(national_tv_games)
""")
