"""
Week 5: Network Games and Price of Anarchy
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

# 1) Pigou network: social cost function and PoA calculation

print("Pigou Network: Price of Anarchy")
print("=" * 55)
print("Route 1: lat = 1 (constant)")
print("Route 2: lat = x (load-dependent)")

x_grid = np.linspace(0, 1, 10000)

def total_cost_pigou(x2):
    """
    Total social cost when fraction x2 uses Route 2
    SC = x1*lat_1 + x2*lat_2 = (1-x2)*1 + x2*x2
    """
    x1 = 1 - x2
    return x1 * 1.0 + x2 * x2

sc_curve = total_cost_pigou(x_grid)

# Social optimum: dSC/dx2 = -1 + 2*x2 = 0 => x2* = 0.5
opt_x2   = 0.5
opt_cost = total_cost_pigou(opt_x2)   # = 0.75

# Nash equilibrium: drivers switch to Route 2 until latencies equal
# lat_1 = 1, lat_2 = x2 => equilibrium when x2 = 1 (all on Route 2, lat=1)
ne_x2   = 1.0
ne_cost = total_cost_pigou(ne_x2)    # = 1.0

poa_pigou = ne_cost / opt_cost

print(f"\nSocial optimum : x2* = {opt_x2:.4f}, SC* = {opt_cost:.4f}")
print(f"Nash equilibrium: x2  = {ne_x2:.4f}, SC  = {ne_cost:.4f}")
print(f"Price of Anarchy: {ne_cost:.4f} / {opt_cost:.4f} = {poa_pigou:.6f}")
print(f"Theoretical PoA (affine latency): 4/3 = {4/3:.6f}")
print(f"Match: {abs(poa_pigou - 4/3) < 1e-6}")

# 2) Braess's Paradox

print("\n" + "=" * 55)
print("Braess's Paradox")
print("=" * 55)

# Original network: S → A → T and S → B → T
# S→A: lat = x_top, A→T: lat = 1
# S→B: lat = 1,     B→T: lat = x_bot
# NE: x_top = x_bot = 0.5, each path has lat = 1.5, SC = 1.5

def original_sc(x_top):
    x_bot   = 1 - x_top
    lat_top = x_top + 1.0     # S->A latency=x_top, A->T latency=1
    lat_bot = 1.0 + x_bot     # S->B latency=1, B->T latency=x_bot
    return x_top * lat_top + x_bot * lat_bot

# Verify NE at x_top=0.5 (latencies equal)
lat_top_ne = 0.5 + 1.0
lat_bot_ne = 1.0 + 0.5
sc_orig_ne = original_sc(0.5)
print(f"Original network NE (50/50 split):")
print(f"  Latency S->A->T = {lat_top_ne:.2f}")
print(f"  Latency S->B->T = {lat_bot_ne:.2f}  (equal => NE)")
print(f"  Social cost = {sc_orig_ne:.3f}")

# After adding A→B (lat=0): dominant path S → A → B → T
# Everyone uses S->A->B->T: x_SA=1, x_BT=1
# Latency S - > A -> B -> T = lat(S->A) + lat(A->B) + lat(B->T) = 1 + 0 + 1 = 2
sc_new_ne = 2.0
print(f"\nWith free A->B link: dominant path S->A->B->T")
print(f"  Latency = 1 (S->A, load=1) + 0 (A->B) + 1 (B->T, load=1) = {sc_new_ne:.2f}")
print(f"  Social cost = {sc_new_ne:.3f}")
print(f"\nBraess PoA = {sc_new_ne:.3f} / {sc_orig_ne:.3f} = {sc_new_ne/sc_orig_ne:.4f}")
print(f"Adding a free road increased latency by {(sc_new_ne/sc_orig_ne - 1)*100:.1f}%")
print(f"Real analogue: Seoul Cheonggyecheon Expressway removal (2003) improved flow")


# 3) HFT latency arms race as congestion game

print("\n" + "=" * 55)
print("HFT Latency Arms Race — Congestion Game Model")
print("=" * 55)

# Model: n symmetric HFT firms, each investing l_i in speed.
# Profit to firm i = alpha * (l_i / sum_j l_j) - c * l_i^2
#
# Symmetric NE (first-order condition, symmetric l_i = l for all i):
#   d/dl_i [alpha * l_i/(n*l) - c*l_i^2] = 0
#   alpha*(n-1)/(n^2 * l^2 / l) -> simplification gives:
#   l_NE = alpha * (n-1) / (2 * c * n^2)   (see Budish et al. 2015 Appendix)
#
# Social optimum: one firm does all investment, others free-ride
# l_social_opt = sqrt(alpha / (2*c)) / n  (distribute evenly but minimally)
# We use simpler: total social investment = alpha/(2*c) at opt vs n*l_NE at NE

ALPHA = 1.0    # total profit pool
C_INV = 0.5   # cost coefficient of speed investment

def hft_ne_per_firm(n):
    #Symmetric NE speed investment per HFT firm
    return ALPHA * (n - 1) / (2 * C_INV * n**2)

def hft_social_opt_per_firm(n):
    """
    Social optimum: minimise total investment while maintaining competition
    Planner can have one firm invest l_opt = sqrt(alpha/(2c)) and rest free-ride
    Per firm allocation = l_opt / n
    """
    l_opt_single = np.sqrt(ALPHA / (2 * C_INV))
    return l_opt_single / n

def hft_total_profit_ne(n):
    #Total industry profit at symmetric NE
    l = hft_ne_per_firm(n)
    # Each firm's profit: alpha/n (equal shares) - c*l^2
    return n * (ALPHA / n - C_INV * l**2)

def hft_total_social_cost_ne(n):
    #Total investment cost at NE (deadweight loss from arms race)
    l = hft_ne_per_firm(n)
    return n * C_INV * l**2

n_arr = np.arange(2, 21)
ne_invest    = np.array([hft_ne_per_firm(n) for n in n_arr])
soc_invest   = np.array([hft_social_opt_per_firm(n) for n in n_arr])
total_ne     = ne_invest * n_arr
social_total = soc_invest * n_arr
waste        = total_ne - social_total   # socially wasteful investment

print(f"\n{'n':>4} {'l_NE/firm':>12} {'l_opt/firm':>12} {'Total NE':>12} {'Total opt':>12} {'Waste':>10}")
print("-" * 65)
for i, n in enumerate(n_arr[:10]):
    print(f"{n:>4} {ne_invest[i]:>12.5f} {soc_invest[i]:>12.5f} "
          f"{total_ne[i]:>12.5f} {social_total[i]:>12.5f} {waste[i]:>10.5f}")

# Key market facts
print("\nKey HFT Market Facts (sourced from literature):")
hft_facts = [
    ("HFT share of US equity order volume (2009 peak)", "~73%"),
    ("HFT share of US equity trading volume (2024)",    "~50-60%"),
    ("Global HFT market value (2024)",                  "$10.36 billion"),
    ("Co location services market",                      ">$600M/yr"),
    ("Chicago NY microwave vs fiber latency saving",    "4 milliseconds"),
    ("Implied cost per millisecond saved (MW towers)",  "$100M+"),
]
for fact, value in hft_facts:
    print(f"  {fact:<52}: {value}")


# 4. Figures

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Top left: Pigou social cost curve
axes[0,0].plot(x_grid, sc_curve, color="steelblue", lw=2, label="Social cost")
axes[0,0].axvline(opt_x2, color="green", linestyle="", lw=1.8,
                  label=f"Optimum (x₂={opt_x2:.2f}, SC={opt_cost:.3f})")
axes[0,0].axvline(ne_x2,  color="red",   linestyle="", lw=1.8,
                  label=f"Nash eq (x₂={ne_x2:.2f}, SC={ne_cost:.3f})")
axes[0,0].scatter([opt_x2], [opt_cost], color="green", s=80, zorder=5)
axes[0,0].scatter([ne_x2],  [ne_cost],  color="red",   s=80, zorder=5)
axes[0,0].annotate(f"PoA = 4/3 ≈ {poa_pigou:.3f}",
                   xy=(0.55, 1.38), fontsize=12, color="darkred", fontweight="bold")
axes[0,0].set_xlabel("Fraction on Route 2"); axes[0,0].set_ylabel("Total social cost")
axes[0,0].set_title("Pigou Network: PoA = 4/3")
axes[0,0].legend(fontsize=9); axes[0,0].grid(alpha=0.3)

# Top right: Braess bar chart
bars = axes[0,1].bar(["Original\n(no A→B link)", "With free\nA→B link"],
                     [sc_orig_ne, sc_new_ne],
                     color=["steelblue", "tomato"], width=0.4, edgecolor="black")
for bar, val in zip(bars, [sc_orig_ne, sc_new_ne]):
    axes[0,1].text(bar.get_x()+bar.get_width()/2, val+0.04, f"{val:.2f}",
                   ha="center", fontweight="bold", fontsize=14)
axes[0,1].set_ylabel("NE total latency"); axes[0,1].set_ylim(0, 2.7)
axes[0,1].set_title("Braess Paradox: Free Road Hurts Everyone")
axes[0,1].grid(axis="y", alpha=0.3)

# Bottom left: HFT per firm investment
axes[1,0].plot(n_arr, ne_invest,  color="tomato",    lw=2, marker="o", label="NE invest/firm")
axes[1,0].plot(n_arr, soc_invest, color="steelblue", lw=2, marker="s", label="Optimal invest/firm")
axes[1,0].set_xlabel("Number of HFT firms")
axes[1,0].set_ylabel("Speed investment per firm")
axes[1,0].set_title("HFT Arms Race: Per Firm Investment")
axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

# Bottom right: Total industry investment
axes[1,1].plot(n_arr, total_ne,     color="tomato",    lw=2, marker="o", label="Total NE invest")
axes[1,1].plot(n_arr, social_total, color="steelblue", lw=2, marker="s", label="Total opt invest")
axes[1,1].fill_between(n_arr, total_ne, social_total, alpha=0.15, color="tomato",
                        label="Wasteful overinvestment")
axes[1,1].set_xlabel("Number of HFT firms")
axes[1,1].set_ylabel("Total industry investment in speed")
axes[1,1].set_title("HFT Arms Race: Aggregate Overinvestment")
axes[1,1].legend(fontsize=9); axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/w5_poa_hft.png", dpi=150)
print("\nSaved: figures/w5_poa_hft.png")
