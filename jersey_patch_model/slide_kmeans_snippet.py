"""
Regenerates the 'K Means Clustering' slide graphic, updated for the two
model changes: total followers -> metro population, and win_pct -> 3-season
average (2023-24, 2024-25, 2025-26), matching nba_kmeans_metro_pop.py.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK = "#0b0b0b"
SUBTITLE_GRAY = "#5a5a5a"
BOX_GRAY = "#4a4a4a"
CODE_COLOR = "#1a1a1a"

fig, ax = plt.subplots(figsize=(17, 9.6), dpi=150, facecolor="white")
ax.set_xlim(0, 17)
ax.set_ylim(0, 9.6)
ax.axis("off")

ax.text(0.9, 8.75, "K Means Clustering", fontsize=34, fontweight="bold", color=INK, va="top")
ax.text(0.9, 7.85,
        "Run K Means to sort the NBA teams (isolating the team strength and exposure components)",
        fontsize=15.5, color=SUBTITLE_GRAY, va="top")

code_lines = [
    "team_matrix = team_avg[",
    '    ["win_pct_3yr_avg", "all_star_count",',
    '     "impressions", "metro_pop",',
    '     "national_tv_games"]]',
    "z = StandardScaler().fit_transform(team_matrix)",
    'team_avg["strength"] = z[:,0] + z[:,1]',
    'team_avg["exposure"] = z[:,2] + z[:,3] + z[:,4]',
    "k = KMeans(n_clusters = 4, n_init = 20).fit(",
    '        team_avg[["strength", "exposure"]])',
    'team_avg["cluster"] = k.labels_',
]
y = 6.65
for line in code_lines:
    ax.text(0.9, y, line, fontsize=15.5, color=CODE_COLOR, family="monospace", va="top")
    y -= 0.565

box = FancyBboxPatch((10.1, 3.65), 6.2, 3.65, boxstyle="round,pad=0,rounding_size=0.08",
                      facecolor=BOX_GRAY, edgecolor="none", zorder=1)
ax.add_patch(box)

ax.text(10.5, 6.83, "Team Strength =", fontsize=19, fontweight="bold", color="white", va="top")
ax.text(10.5, 6.23, "z(win_pct_3yr_avg) + z(all_star_count)", fontsize=16.5, color="white", va="top")

ax.text(10.5, 5.30, "Exposure =", fontsize=19, fontweight="bold", color="white", va="top")
ax.text(10.5, 4.70, "z(impressions) + z(metro_pop)", fontsize=16.5, color="white", va="top")
ax.text(10.5, 4.27, "+ z(national_tv_games)", fontsize=16.5, color="white", va="top")

fig.savefig("slide_kmeans_snippet.png", facecolor="white", bbox_inches=None)
plt.close(fig)
print("Saved slide_kmeans_snippet.png")
