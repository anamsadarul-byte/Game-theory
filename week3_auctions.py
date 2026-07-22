"""
Week 3: Auctions and Market Design
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# 1)Second price (Vickrey) auction

def simulate_second_price(n_bidders, n_trials, v_low=0, v_high=100):
    """
    Vickrey auction: bidding true value is a weakly dominant strategy
    Winner = highest bidder; pays second highest bid

    Expected revenue (analytic): E[2nd order stat of n Uniform(0,H)]
      = H * (n-1) / (n+1)

    Returns array of seller revenues
    """
    revenues = np.zeros(n_trials)
    for t in range(n_trials):
        values = np.random.uniform(v_low, v_high, n_bidders)
        sorted_v = np.sort(values)[::-1]
        revenues[t] = sorted_v[1]          # second-highest = truthful second price
    return revenues

# 2)First price sealed bid auction

def simulate_first_price(n_bidders, n_trials, v_low=0, v_high=100):
    """
    First-price auction: symmetric BNE bidding strategy under Uniform[0,H] values:
        b(v) = v * (n-1)/n

    This is derived from the first-order condition on the bidder's expected profit:
        max_{b} (v - b) * Pr[b wins]
      = max_{b} (v - b) * (b / H)^{n-1}   [all others bid below b with prob (b/H)^{n-1}]
    FOC: -(b/H)^{n-1} + (v-b)*(n-1)/H * (b/H)^{n-2} / H = 0
    Solving: b*(v) = v * (n-1)/n

    Winner = highest bidder; pays their own (shaded) bid
    Returns array of seller revenues
    """
    shading_factor = (n_bidders - 1) / n_bidders
    revenues = np.zeros(n_trials)
    for t in range(n_trials):
        values = np.random.uniform(v_low, v_high, n_bidders)
        bids = values * shading_factor
        revenues[t] = np.max(bids)
    return revenues

# 3)Analytic Revenue Equivalence check

def analytic_revenue(n_bidders, v_high=100):
    """
    Under U[0,H] values and n symmetric bidders:
      E[seller revenue] = H * (n-1) / (n+1)
    This holds for BOTH first and second price auctions (Revenue Equivalence).
    """
    return v_high * (n_bidders - 1) / (n_bidders + 1)

# 4)Main experiment

N_BIDDERS = 5
N_TRIALS  = 10_000
V_HIGH    = 100

sp_rev = simulate_second_price(N_BIDDERS, N_TRIALS, v_high=V_HIGH)
fp_rev = simulate_first_price(N_BIDDERS, N_TRIALS, v_high=V_HIGH)
analytic = analytic_revenue(N_BIDDERS, V_HIGH)

print("Revenue Equivalence Theorem: Numerical Verification")
print("=" * 58)
print(f"Bidders : {N_BIDDERS} ,  Trials: {N_TRIALS:,} , Values ~ U[0,{V_HIGH}]")
print(f"Analytic revenue (both formats): {analytic:.4f}")
print()
print(f"{'Format':<32} {'Simulated Mean':>15} {'Std Dev':>10} {'Error vs Theory':>16}")
print("" * 75)
print(f"{'Second price (truthful bidding)':<32} {sp_rev.mean():>15.4f} "
      f"{sp_rev.std():>10.4f} {abs(sp_rev.mean()-analytic):>16.4f}")
print(f"{'First-price (b=v*(n-1)/n)':<32} {fp_rev.mean():>15.4f} "
      f"{fp_rev.std():>10.4f} {abs(fp_rev.mean()-analytic):>16.4f}")
print(f"\nGap between formats: {abs(fp_rev.mean() - sp_rev.mean()):.4f}  (≈0 confirms RET)")
print(f"\nVariance comparison:")
print(f"  Second price std: {sp_rev.std():.3f}  (higher: truthful bids spread widely)")
print(f"  First price  std: {fp_rev.std():.3f}  (lower: shading compresses distribution)")

# 5) Bid shading vs. number of bidders

print("\nBid Shading by Number of Bidders (first price auction):")
print(f"{'n':>5} {'Shade factor b(v)/v':>22} {'Shading %':>12} {'Analytic revenue':>18}")
print("" * 60)
bidder_counts = np.arange(2, 16)
for n in bidder_counts:
    sf  = (n - 1) / n
    sh  = (1 - sf) * 100
    rev = analytic_revenue(n, V_HIGH)
    print(f"{n:>5} {sf:>22.4f} {sh:>11.1f}% {rev:>18.3f}")

# 6) Real world auction benchmarks

print("\n" + "=" * 58)
print("Real World Auction Mechanisms")
print("=" * 58)

auctions = [
    ("US Treasury T bills",
     "Discriminatory (1st price multi-unit)",
     "~$20–30 trillion/yr",
     "Winner's curse risk; discriminatory format"),
    ("US Treasury T-notes/bonds",
     "Uniform price (~2nd price multi-unit)",
     "~$10 trillion/yr",
     "Uniform-price reduces bid shading incentive"),
    ("FCC AWS-3 Spectrum (2015)",
     "SMRA (≈ ascending Vickrey multi-item)",
     "$41.33 billion, 31 bidders, 1,611 licenses",
     "Largest US spectrum auction; $2.72/MHz-pop"),
    ("Google IPO (2004)",
     "Dutch/Uniform price",
     "$1.67B at $85/share clearing price",
     "Reduced underpricing vs. traditional book-build"),
    ("Google/eBay Ad Slots",
     "Generalised Second Price (GSP)",
     ">$100B/yr globally",
     "GSP ≠ VCG; equilibrium bidding more complex"),
]

for name, fmt, revenue, note in auctions:
    print(f"\n{name}")
    print(f"  Format  : {fmt}")
    print(f"  Revenue : {revenue}")
    print(f"  Note    : {note}")

# 7. Figures

shading_vals = (1 - (bidder_counts - 1) / bidder_counts) * 100
analytic_revs = np.array([analytic_revenue(n, V_HIGH) for n in bidder_counts])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(sp_rev, bins=45, alpha=0.6, color="steelblue",
             label=f"Second price  μ={sp_rev.mean():.2f}, σ={sp_rev.std():.2f}")
axes[0].hist(fp_rev, bins=45, alpha=0.6, color="darkorange",
             label=f"First price   μ={fp_rev.mean():.2f}, σ={fp_rev.std():.2f}")
axes[0].axvline(sp_rev.mean(),  color="steelblue",  linestyle="", lw=1.8)
axes[0].axvline(fp_rev.mean(),  color="darkorange", linestyle="", lw=1.8)
axes[0].axvline(analytic,        color="black",       linestyle=":",  lw=1.5,
                label=f"Analytic: {analytic:.2f}")
axes[0].set_xlabel("Seller Revenue")
axes[0].set_ylabel("Frequency")
axes[0].set_title(f"Revenue Equivalence ({N_BIDDERS} bidders, {N_TRIALS:,} trials)")
axes[0].legend(fontsize=9)

ax2 = axes[1]
color1, color2 = "darkorange", "steelblue"
ln1 = ax2.plot(bidder_counts, shading_vals,  color=color1, marker="o", lw=2,
               label="Bid shading (% below value)")
ax2b = ax2.twinx()
ln2 = ax2b.plot(bidder_counts, analytic_revs, color=color2, marker="s", lw=2,
                linestyle="", label="Expected revenue")
ax2.set_xlabel("Number of Bidders")
ax2.set_ylabel("Bid shading (%)", color=color1)
ax2b.set_ylabel("Expected seller revenue", color=color2)
ax2.set_title("Bid Shading and Revenue vs. Competition (1st price)")
lines = ln1 + ln2
ax2.legend(lines, [l.get_label() for l in lines], fontsize=9)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/w3_auctions.png", dpi=150)
print("\nSaved: figures/w3_auctions.png")
