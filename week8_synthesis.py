"""
Week 8: Synthesis: Price of Anarchy Across All Settings
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

np.random.seed(42)

# 1) PoA derivations (self contained)

print("Price of Anarchy: Analytical Derivations")
print("=" * 60)

#Pigou network
opt_cost_pigou = 0.5 * 1 + 0.5 * 0.5    # x1=0.5, x2=0.5: 0.5*1 + 0.5*0.5 = 0.75
ne_cost_pigou  = 0.0 * 1 + 1.0 * 1.0    # x2=1: all on Route 2, lat=1
poa_pigou      = ne_cost_pigou / opt_cost_pigou
print(f"\n1 Pigou network:")
print(f"   Opt cost  = {opt_cost_pigou:.4f}  (x2*=0.5, SC=x1*1+x2*x2=0.5+0.25)")
print(f"   Nash cost = {ne_cost_pigou:.4f}  (x2=1, SC=1*1=1.0)")
print(f"   PoA = {poa_pigou:.6f} = 4/3 = {4/3:.6f}  [tight bound for affine latency]")

# Braess paradox
ne_cost_orig  = 1.5   # 50/50 split, each path lat=1.5
ne_cost_braess = 2.0  # all on S -> A -> B -> T, lat=2
poa_braess = ne_cost_braess / ne_cost_orig
print(f"\n2 Braess paradox:")
print(f"   NE cost before adding link: {ne_cost_orig:.4f}")
print(f"   NE cost after free link:    {ne_cost_braess:.4f}")
print(f"   PoA (worsening ratio): {poa_braess:.6f}")

#HFT arms race: PoA grows without bound
print(f"\n3 HFT latency arms race:")
print(f"   Per firm NE investment: l_NE = alpha*(n-1)/(2*c*n^2)")
print(f"   Total NE investment: n*l_NE = alpha*(n-1)/(2*c*n)")
print(f"   Social optimum: l_opt = sqrt(alpha/(2c)) / n (one firm invests, rest free ride)")
print(f"   PoA grows O(n): unbounded as number of HFT firms increases")
alpha, c_inv = 1.0, 0.5
for n in [2, 5, 10, 20]:
    l_ne  = alpha*(n-1)/(2*c_inv*n**2)
    total_ne_inv = n * l_ne
    l_opt_single = np.sqrt(alpha/(2*c_inv))
    social_total = l_opt_single   # one firm does it all
    poa_n = total_ne_inv / social_total if social_total > 0 else np.inf
    print(f"   n={n:>3}: Total NE={total_ne_inv:.4f}, Soc opt={social_total:.4f}, ratio={poa_n:.4f}")

# Bertrand duopoly: price = MC => social optimum => PoA = 1
print(f"\n4 Bertrand duopoly:")
print(f"   NE: price = marginal cost => P* = c, zero profit")
print(f"   Social optimum also requires P = c (consumer surplus maximised)")
print(f"   PoA = 1.000 exactly (Bertrand paradox = social optimum)")

# Revenue Equivalence => auction PoA ~ 1
sp = np.array([np.sort(np.random.uniform(0,100,5))[-2] for _ in range(10000)])
fp = np.array([np.max(np.random.uniform(0,100,5)*(4/5)) for _ in range(10000)])
poa_auction = fp.mean() / sp.mean()   # first price / second price
print(f"\n5 First price auction (vs. second price efficient benchmark):")
print(f"   FP mean revenue: {fp.mean():.3f}")
print(f"   SP mean revenue: {sp.mean():.3f}")
print(f"   PoA = {poa_auction:.6f}  (≈1 by Revenue Equivalence Theorem)")

# Hawk Dove ESS: social welfare at ESS vs. all Dove optimum
V, C = 4, 6
p_ess = V / C
def hd_avg_fitness(p, V, C):
    #Population average fitness in Hawk-Dove game at hawk frequency p
    hawk_fit = p*(V-C)/2 + (1-p)*V
    dove_fit = p*0 + (1-p)*V/2
    return p*hawk_fit + (1-p)*dove_fit
welfare_ess  = hd_avg_fitness(p_ess, V, C)
welfare_opt  = hd_avg_fitness(0.0,   V, C)   # all Dove maximises welfare
poa_hd = welfare_opt / welfare_ess if welfare_ess > 0 else 1.0
print(f"\n6 Hawk Dove ESS:")
print(f"   ESS hawk freq: {p_ess:.4f}")
print(f"   Avg welfare at ESS:     {welfare_ess:.4f}")
print(f"   Avg welfare (all-Dove): {welfare_opt:.4f}")
print(f"   PoA = {poa_hd:.6f}")

# 2)Comprehensive PoA summary table

print("\n" + "=" * 60)
print("Price of Anarchy: Summary Table")
print("=" * 60)

settings = [
    ("Pigou Network",           poa_pigou,    "Network routing",    "Tight 4/3 bound; Roughgarden-Tardos (2002)"),
    ("Braess Paradox",          poa_braess,   "Network routing",    "Adding free road worsens NE by 33%"),
    ("HFT Arms Race (n=10)",    None,         "Finance/Congestion", "PoA → ∞ as n grows; see Budish et al"),
    ("Bertrand Duopoly",        1.0,          "Market competition", "Bertrand paradox = socially efficient NE"),
    ("1st-Price Auction",       poa_auction,  "Market mechanism",   "RET: ≈1; exact equality in continuous limit"),
    ("Hawk-Dove ESS",           poa_hd,       "Evolutionary",       "ESS is globally stable but not welfare max"),
]

print(f"\n{'Setting':<28} {'PoA':>8}  {'Category':<22} {'Notes'}")
print("-" * 100)
for name, poa, cat, note in settings:
    poa_str = f"{poa:.4f}" if poa is not None else "→ ∞"
    print(f"{name:<28} {poa_str:>8}  {cat:<22} {note}")

# 3)Financial market efficiency comparison

print("\n" + "=" * 60)
print("Financial Markets as Games: Efficiency Summary")
print("=" * 60)

market_data = [
    ("OPEC / Oil market",          "Cournot",     "~1.3–1.5",  "Cartel restricts output; PoA from monopoly inefficiency"),
    ("Equity order book",          "Double auction", "~1.0",   "Competitive; bid ask spread ≈ adverse selection cost"),
    ("IPO pricing (bookbuild)",    "1st price",   "~1.05–1.2", "Historical underpricing 1520%; mechanism design gap"),
    ("Spectrum auctions (FCC)",    "SMRA/VCG",    "~1.0–1.05", "SMRA reduces exposure problem; near efficient"),
    ("HFT co location race",       "Congestion",  ">> 1",      "Latency investment is socially wasteful arms race"),
    ("Ad slot auctions (Google)",  "GSP",         "~1.0–1.1",  "GSP ≠ VCG; small but nonzero efficiency loss"),
]

print(f"\n{'Market':<30} {'Mechanism':<16} {'PoA (approx)':>14}  Notes")
print("-" * 90)
for market, mech, poa, note in market_data:
    print(f"{market:<30} {mech:<16} {poa:>14}  {note}")

# 4. Figures

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: PoA bar chart (excluding HFT which is infinite)
names_plot  = ["Pigou\nNetwork", "Braess\nParadox", "Bertrand\nDuopoly",
               "1st-Price\nAuction", "Hawk Dove\nESS"]
poas_plot   = [poa_pigou, poa_braess, 1.0, poa_auction, poa_hd]
colors_plot = ["tomato", "tomato", "steelblue", "steelblue", "darkorange"]

bars = axes[0].bar(names_plot, poas_plot, color=colors_plot,
                   edgecolor="black", width=0.5)
axes[0].axhline(1.0, color="black", linestyle="-", alpha=0.6,
                label="PoA = 1 (no efficiency loss)")
for bar, val in zip(bars, poas_plot):
    axes[0].text(bar.get_x()+bar.get_width()/2, val+0.01,
                 f"{val:.3f}", ha="center", fontweight="bold", fontsize=11)
legend_patches = [
    mpatches.Patch(color="tomato",     label="Network routing"),
    mpatches.Patch(color="steelblue",  label="Market mechanisms"),
    mpatches.Patch(color="darkorange", label="Evolutionary"),
]
axes[0].legend(handles=legend_patches, fontsize=9)
axes[0].set_ylabel("Price of Anarchy")
axes[0].set_title("Price of Anarchy Across All Game Settings")
axes[0].set_ylim(0, 1.7); axes[0].grid(axis="y", alpha=0.3)

# Right: HFT PoA growth
n_arr = np.arange(2, 31)
alpha_hft, c_hft = 1.0, 0.5
l_opt_single_hft = np.sqrt(alpha_hft / (2*c_hft))
total_ne_hft  = np.array([n * alpha_hft*(n-1)/(2*c_hft*n**2) for n in n_arr])
poa_hft_arr   = total_ne_hft / l_opt_single_hft

axes[1].plot(n_arr, poa_hft_arr, color="tomato", lw=2.5, marker="o", markersize=4)
axes[1].axhline(1.0, color="black", linestyle="-", alpha=0.6, label="PoA = 1 (efficient)")
axes[1].set_xlabel("Number of HFT firms")
axes[1].set_ylabel("Price of Anarchy (total NE invest / social opt)")
axes[1].set_title("HFT Latency Arms Race: PoA Grows Without Bound")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/w8_synthesis.png", dpi=150)
print("\nSaved: figures/w8_synthesis.png")
