## NBA Jersey Patch Valuation - Summary

**Regression** (fit on 2026-equivalent $; calibration blends current-deal and time-adjusted historical prices -- see Step 2 for why year isn't a direct regression feature):
- Value(2026-equiv) = 31,099,672 + 0.0644 x Exposure_Value_Zoomph
- R² = 0.014  |  n = 11 calibration points (4 current + 9 historical, de-duplicated)
- Growth rate used to time-adjust historical prices to 2026-equivalent $: 10.2%/yr (from Lakers + Warriors trajectories)

**Market-size tiers**

| Tier | Teams | Avg Value | Median Value |
|---|---|---|---|
| Large | 13 | $32,006,196 | $32,793,616 |
| Mid | 13 | $32,692,742 | $32,484,455 |
| Small | 4 | $33,860,633 | $31,998,170 |

**All 30 teams, sorted by value descending** (value is real price where confirmed, else the model's 2026-equivalent estimate):

| Team | Value ($) | Confidence | Tier |
|---|---|---|---|
| Golden State Warriors | $50,000,000 | Real | Large |
| Oklahoma City Thunder | $40,000,000 | Real | Small |
| San Antonio Spurs | $35,962,515 | Estimated | Mid |
| Boston Celtics | $33,740,421 | Estimated | Large |
| Minnesota Timberwolves | $33,353,970 | Estimated | Mid |
| Denver Nuggets | $33,328,206 | Estimated | Mid |
| Houston Rockets | $33,308,884 | Estimated | Large |
| Dallas Mavericks | $33,102,777 | Estimated | Large |
| Phoenix Suns | $33,025,486 | Estimated | Mid |
| Chicago Bulls | $32,922,433 | Estimated | Large |
| New York Knicks | $32,800,057 | Estimated | Large |
| LA Clippers | $32,793,616 | Estimated | Large |
| Orlando Magic | $32,697,003 | Estimated | Mid |
| Miami Heat | $32,593,949 | Estimated | Mid |
| Detroit Pistons | $32,484,455 | Estimated | Mid |
| Cleveland Cavaliers | $32,420,046 | Estimated | Mid |
| Milwaukee Bucks | $32,239,702 | Estimated | Small |
| Indiana Pacers | $32,181,735 | Estimated | Mid |
| Atlanta Hawks | $32,033,595 | Estimated | Large |
| Portland Trail Blazers | $32,007,832 | Estimated | Mid |
| Brooklyn Nets | $32,007,832 | Estimated | Large |
| Charlotte Hornets | $31,885,456 | Estimated | Mid |
| Toronto Raptors | $31,866,133 | Estimated | Large |
| New Orleans Pelicans | $31,756,638 | Estimated | Small |
| Sacramento Kings | $31,631,042 | Estimated | Mid |
| Washington Wizards | $31,504,801 | Estimated | Large |
| Memphis Grizzlies | $31,446,189 | Estimated | Small |
| Utah Jazz | $31,433,952 | Estimated | Mid |
| LA Lakers | $30,000,000 | Real | Large |
| Philadelphia 76ers | $10,000,000 | Real | Large |
