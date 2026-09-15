"""HW3 Problem 6: Benford's Law analysis of US county populations (2010)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- (a) Verify P(D=j) = log10((j+1)/j) is a valid PMF ---
digits = np.arange(1, 10)
benford_probs = np.log10((digits + 1) / digits)
print("Benford probabilities:")
for j, p in zip(digits, benford_probs):
    print(f"  P(D={j}) = {p:.4f}")
print(f"Sum = {benford_probs.sum():.10f}  (should equal 1)")
print(f"All probabilities >= 0: {np.all(benford_probs >= 0)}")
print()

# --- (b) Leading digit histogram of population_2010 ---
df = pd.read_csv("us_county_population_2010.csv")
leading_digit = df["population_2010"].astype(str).str[0].astype(int)

observed_counts = leading_digit.value_counts().reindex(digits, fill_value=0)
observed_freq = observed_counts / observed_counts.sum()

fig, ax = plt.subplots(figsize=(8, 5))
x = digits
width = 0.4
ax.bar(x - width/2, observed_freq.values, width=width, label="Observed (county population)")
ax.bar(x + width/2, benford_probs, width=width, label="Benford's Law")
ax.set_xlabel("Leading digit")
ax.set_ylabel("Relative frequency")
ax.set_title("Leading digit of 2010 US county populations vs. Benford's Law")
ax.set_xticks(digits)
ax.legend()
fig.tight_layout()
fig.savefig("benford_histogram.png", dpi=150)
print("Saved plot to benford_histogram.png")
print()
print("Observed vs Benford frequencies:")
comparison = pd.DataFrame({
    "observed_freq": observed_freq.values,
    "benford_freq": benford_probs,
    "diff": observed_freq.values - benford_probs,
})
comparison.index = digits
print(comparison)
