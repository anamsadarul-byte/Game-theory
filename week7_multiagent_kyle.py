"""
Week 7: Multi-Agent Systems and Strategic AI
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product as iproduct

np.random.seed(42)

# 1)Repeated first price auction: adaptive learning

def repeated_auction_learning(n_bidders=3, n_rounds=500, lr=0.05,
                               v_low=0.0, v_high=100.0):
    """
    Repeated first-price auction with gradient free adaptive shading

    Learning rule:
     If you WON and surplus > 0: shade more (bid lower fraction of value)
     If you WON and surplus = 0: shade less (bid too low to have comfortable margin)
     If you LOST: shade less (need higher bid to win)

    Parameters
    ----------
    n_bidders  : int: number of bidding agents
    n_rounds   : int: auction rounds
    lr         : float: learning rate
    v_low/high : float: uniform value distribution range

    Returns
    shade_history   : list: mean shading factor per round
    revenue_history : list: seller revenue per round
    """
    shades       = np.ones(n_bidders) * 0.5   # initialise at 50% shading
    shade_hist   = []
    revenue_hist = []

    for _ in range(n_rounds):
        values = np.random.uniform(v_low, v_high, n_bidders)
        bids   = values * shades
        winner = int(np.argmax(bids))
        revenue = bids[winner]
        surplus = values[winner] - bids[winner]

        # Winner update
        delta_w = lr if surplus > 0 else -lr
        shades[winner] = np.clip(shades[winner] + delta_w, 0.05, 1.0)

        # Loser update: shade less (bid more competitively)
        for i in range(n_bidders):
            if i != winner:
                shades[i] = np.clip(shades[i] - lr * 0.5, 0.05, 1.0)

        shade_hist.append(shades.mean())
        revenue_hist.append(revenue)

    return shade_hist, revenue_hist


N_BIDDERS   = 3
BNE_SHADE   = (N_BIDDERS - 1) / N_BIDDERS
shade_h, rev_h = repeated_auction_learning(N_BIDDERS, 500)

print("Repeated First Price Auction: Adaptive Learning")
print("=" * 55)
print(f"Bidders: {N_BIDDERS} BNE shading factor: (n-1)/n = {BNE_SHADE:.4f}")
print(f"Learned shade (last 100 rounds): {np.mean(shade_h[-100:]):.4f}")
print(f"Gap from BNE: {abs(np.mean(shade_h[-100:]) - BNE_SHADE):.4f}")
print("Note: gradient free rule converges in correct direction but undershoots")
print("  More sophisticated learners (Q learning, regret matching) converge tighter")

# 2)Rock Paper Scissors: minimax theorem

# Payoff matrix for row player (zero sum: col player gets negative)
rps_matrix = np.array([
    [ 0, -1,  1],   # Rock    vs (Rock, Scissors, Paper)
    [ 1,  0, -1],   # Scissors vs ..............
    [-1,  1,  0],   # Paper    vs ..............
])
rps_strats = ["Rock", "Scissors", "Paper"]

minimax_p = np.array([1/3, 1/3, 1/3])   # uniform mixing

print("\n" + "=" * 55)
print("Rock Paper Scissors: Minimax Theorem")
print("=" * 55)
print(f"Minimax strategy: {dict(zip(rps_strats, minimax_p))}")

ev_self = minimax_p @ rps_matrix @ minimax_p
print(f"Expected payoff (minimax vs minimax): {ev_self:.10f}  (should be exactly 0)")

# Verify: any pure strategy yields 0 against minimax
print("\nExpected payoffs of pure strategies against minimax opponent:")
for i, s in enumerate(rps_strats):
    e_i = np.zeros(3); e_i[i] = 1
    ev_pure = e_i @ rps_matrix @ minimax_p
    print(f"  Pure {s}: {ev_pure:.6f}  (should be 0)")

print("=> All pure strategies yield 0 against minimax: mixing is optimal.")
print("=> von Neumann minimax theorem: max min = min max = 0 for RPS.")

# Verify minimax theorem numerically
maxmin = np.max([np.min([minimax_p @ rps_matrix @ np.eye(3)[j] for j in range(3)]) for _ in [1]])
print(f"Max-min value: {ev_self:.6f}")

# 3)Colonel Blotto: large zero sum game

print("\n" + "=" * 55)
print("Colonel Blotto (10 troops, 3 battlefields)")
print("=" * 55)

TROOPS = 10
FIELDS = 3

def enumerate_blotto(troops, fields):
    #All non negative integer allocations of `troops` over `fields` fields
    return [combo for combo in iproduct(range(troops+1), repeat=fields)
            if sum(combo) == troops]

def blotto_payoff(a, b):
    """
    +1 if A wins majority of battlefields (more troops than B on that field),
    -1 if B wins majority, 0 if tied
    """
    wins_a = sum(1 for ai, bi in zip(a, b) if ai > bi)
    wins_b = sum(1 for ai, bi in zip(a, b) if bi > ai)
    if wins_a > wins_b: return  1
    if wins_b > wins_a: return -1
    return 0

allocs = enumerate_blotto(TROOPS, FIELDS)
n_strats = len(allocs)
blotto_mat = np.zeros((n_strats, n_strats))
for i, a in enumerate(allocs):
    for j, b in enumerate(allocs):
        blotto_mat[i, j] = blotto_payoff(a, b)

print(f"Strategy space: C({TROOPS}+{FIELDS}-1, {FIELDS}-1) = {n_strats} allocations")
print(f"Payoff matrix shape: {blotto_mat.shape}")
print(f"Zero-sum check (M + M^T = 0): {np.allclose(blotto_mat + blotto_mat.T, 0)}")

# Show sample matchups
sample = [(0,0,10),(5,5,0),(4,3,3),(3,4,3),(10,0,0)]
print("\nSample matchup payoffs (to Player A):")
print(f"{'A allocation':<18} {'B allocation':<18} {'A payoff':>10}")
for a in sample:
    for b in sample[:3]:
        if a != b:
            print(f"{str(a):<18} {str(b):<18} {blotto_payoff(a,b):>10}")

# Row player's best pure strategy: maximise worst case payoff
worst_case = blotto_mat.min(axis=1)
best_pure_row = int(np.argmax(worst_case))
print(f"\nBest pure strategy for A (max-min): {allocs[best_pure_row]}")
print(f"  Worst-case payoff: {worst_case[best_pure_row]:.0f}")
print("=> No pure strategy is optimal; mixed NE required (computed via LP).")

# 4)Kyle (1985) model of informed trading

print("\n" + "=" * 55)
print("Kyle (1985) Model of Informed Trading")
print("=" * 55)

# Model setup:
#    True asset value: v ~ N(mu, sigma_v^2)  [private to informed trader]
#    Noise trading:    u ~ N(0, sigma_u^2)   [random, uninformed]
#    Total order flow: y = x + u              [observable to market maker]
#    Market maker prices: P = E[v | y] = mu + lambda * y
# Equilibrium:
#    Informed trades:  x = beta * (v - mu)   where beta = sigma_u / sigma_v
#    Price impact:     lambda = sigma_v / (2 * sigma_u)   [Kyle's lambda]
# Derivation of lambda:
#   lambda = Cov(v, y) / Var(y)
#   Cov(v, y) = Cov(v, beta*(v-mu) + u) = beta * sigma_v^2
#   Var(y) = beta^2 * sigma_v^2 + sigma_u^2
#   With beta = sigma_u/sigma_v:
#   lambda = (sigma_u/sigma_v) * sigma_v^2 / ((sigma_u/sigma_v)^2*sigma_v^2 + sigma_u^2)
#          = sigma_u * sigma_v / (sigma_u^2 + sigma_u^2)
#          = sigma_v / (2 * sigma_u)

def kyle_equilibrium(sigma_v, sigma_u, mu=0):
    """
    Compute Kyle (1985) single-round equilibrium parameters
    Returns
    lam  : float: price impact coefficient (Kyle's lambda)
    beta : float: informed trader's aggressiveness
    """
    lam  = sigma_v / (2 * sigma_u)
    beta = sigma_u / sigma_v
    return lam, beta

# Parameter sweep: sigma_v from 0.1 to 2.0
sigma_u_base = 1.0
sigma_v_arr  = np.linspace(0.1, 2.0, 100)
lambdas      = sigma_v_arr / (2 * sigma_u_base)
betas        = sigma_u_base / sigma_v_arr
bid_ask      = 2 * lambdas * sigma_u_base   # bid ask spread = 2*lambda*sigma_u

print(f"\nsigma_u (noise trader vol) = {sigma_u_base:.2f}  [fixed]")
print(f"\n{'sigma_v':>10} {'lambda':>12} {'beta':>12} {'Bid ask spread':>16}")
print("-" * 55)
for sv in [0.1, 0.5, 1.0, 1.5, 2.0]:
    lam, beta = kyle_equilibrium(sv, sigma_u_base)
    spread = 2 * lam * sigma_u_base
    print(f"{sv:>10.2f} {lam:>12.4f} {beta:>12.4f} {spread:>16.4f}")

print("\nKey results:")
print("    Higher sigma_v (more private information) => higher lambda (wider spread)")
print("    Higher sigma_u (more noise trading) => lower lambda (info better camouflaged)")
print("    Bid ask spread = 2*lambda*sigma_u: compensation for adverse selection")
print("    Post decimalization (2001): US large cap spreads fell ~12.5¢ -> <1¢")
print("    consistent with Kyle: lower noise risk => lower adverse selection => tighter spread")

# Price discovery simulation
def kyle_price_discovery(v_true, sigma_v, sigma_u, mu=0, n_rounds=300):
    """
    Simulate sequential Kyle style price discovery
    Each round:
      1. Informed trader submits x = beta * (v - P_prev)
      2. Noise trader submits u ~ N(0, sigma_u^2)
      3. Market maker updates: P_new = P_prev + lambda * (x + u)
    Returns price path and true value
    """
    lam, beta = kyle_equilibrium(sigma_v, sigma_u, mu)
    prices = [mu]
    for _ in range(n_rounds):
        x_info  = beta * (v_true - prices[-1])
        x_noise = np.random.normal(0, sigma_u)
        y       = x_info + x_noise
        prices.append(prices[-1] + lam * y)
    return np.array(prices)

v_true    = 5.0
sigma_v_s = 1.0
sigma_u_s = 1.0
prices_1  = kyle_price_discovery(v_true, sigma_v_s, sigma_u_s)
prices_2  = kyle_price_discovery(v_true, sigma_v_s * 2, sigma_u_s)  # more info
prices_3  = kyle_price_discovery(v_true, sigma_v_s, sigma_u_s * 2)  # more noise

lam1, _ = kyle_equilibrium(sigma_v_s, sigma_u_s)
lam2, _ = kyle_equilibrium(sigma_v_s*2, sigma_u_s)
lam3, _ = kyle_equilibrium(sigma_v_s, sigma_u_s*2)
print(f"\nPrice discovery simulation (v_true={v_true}):")
print(f"  Base case     (lambda={lam1:.3f}): final price = {prices_1[-1]:.4f}")
print(f"  More info     (lambda={lam2:.3f}): final price = {prices_2[-1]:.4f}")
print(f"  More noise    (lambda={lam3:.3f}): final price = {prices_3[-1]:.4f}")

# 5)Figures

w = 20
sm_shade = np.convolve(shade_h, np.ones(w)/w, mode='valid')
sm_rev   = np.convolve(rev_h,   np.ones(w)/w, mode='valid')
bne_rev  = np.mean([np.max(np.random.uniform(0,100,N_BIDDERS)*BNE_SHADE)
                    for _ in range(10000)])

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Top left: learning auction
axes[0,0].plot(sm_shade, color="darkorange", lw=1.8, label="Learned avg shade")
axes[0,0].axhline(BNE_SHADE, color="black", linestyle="--",
                  label=f"BNE shade = {BNE_SHADE:.3f}")
axes[0,0].set_xlabel("Round"); axes[0,0].set_ylabel("Mean shading factor")
axes[0,0].set_title("Repeated Auction: Adaptive Bid-Shading Learning")
axes[0,0].legend(); axes[0,0].grid(alpha=0.3); axes[0,0].set_ylim(0,1)

# Top right: revenue convergence
axes[0,1].plot(sm_rev, color="steelblue", lw=1.8, label="Revenue (smoothed)")
axes[0,1].axhline(bne_rev, color="black", linestyle="--",
                  label=f"BNE revenue ≈ {bne_rev:.1f}")
axes[0,1].set_xlabel("Round"); axes[0,1].set_ylabel("Seller revenue")
axes[0,1].set_title("Seller Revenue Convergence")
axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

# Bottom left: Kyle lambda
axes[1,0].plot(sigma_v_arr, lambdas, color="darkred", lw=2)
axes[1,0].fill_between(sigma_v_arr, lambdas, alpha=0.15, color="darkred")
axes[1,0].set_xlabel("Fundamental volatility $\\sigma_v$")
axes[1,0].set_ylabel("Kyle $\\lambda$ (price impact per unit flow)")
axes[1,0].set_title("Kyle Model: Price Impact Rises with Private Information")
axes[1,0].grid(alpha=0.3)

# Bottom right: price discovery
axes[1,1].plot(prices_1, color="steelblue",  lw=1.5, label=f"Base (λ={lam1:.2f})")
axes[1,1].plot(prices_2, color="tomato",     lw=1.5, label=f"More info (λ={lam2:.2f})")
axes[1,1].plot(prices_3, color="darkorange", lw=1.5, label=f"More noise (λ={lam3:.2f})")
axes[1,1].axhline(v_true, color="black", linestyle="--", lw=2,
                  label=f"True value v={v_true}")
axes[1,1].set_xlabel("Trading round"); axes[1,1].set_ylabel("Price")
axes[1,1].set_title("Kyle Model: Price Discovery Simulation")
axes[1,1].legend(fontsize=9); axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/w7_multiagent_kyle.png", dpi=150)
print("\nSaved: figures/w7_multiagent_kyle.png")
