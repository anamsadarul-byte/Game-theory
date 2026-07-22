"""
Week 2: Nash Equilibrium and Strategic Stability
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1)Pure strategy Nash equilibrium finder

def find_pure_nash(payoff_A, payoff_B):
    """
    Find all pure strategy Nash equilibria for a 2 player normal form game.

    A profile (i,j) is a Nash equilibrium iff:
      payoff_A[i,j] >= payoff_A[k,j]  for all k  (A can't profitably deviate)
      payoff_B[i,j] >= payoff_B[i,k]  for all k  (B can't profitably deviate)

    Parameters
    payoff_A : ndarray (m, n)   row player payoffs
    payoff_B : ndarray (m, n)   column player payoffs

    Returns
    list of (int, int) (row, col) pure NE indices
    """
    m, n = payoff_A.shape
    equilibria = []
    for i in range(m):
        for j in range(n):
            a_no_deviate = np.all(payoff_A[i, j] >= payoff_A[:, j])
            b_no_deviate = np.all(payoff_B[i, j] >= payoff_B[i, :])
            if a_no_deviate and b_no_deviate:
                equilibria.append((i, j))
    return equilibria

# 2)Mixed-strategy NE for 2x2 games

def find_mixed_nash_2x2(payoff_A, payoff_B):
    """
    Find the mixed strategy Nash equilibrium for a 2x2 game

    Method: indifference conditions
      Player A mixes with prob p on row 0: make B indifferent between cols
      Player B mixes with prob q on col 0: make A indifferent between rows

    Returns (p, q) if a valid interior mixed NE exists, else None
    """
    # B's indifference: col 0 vs col 1 given A plays row 0 with prob p
    # p*pB[0,0] + (1-p)*pB[1,0] = p*pB[0,1] + (1-p)*pB[1,1]
    a, b = payoff_B[0, 0], payoff_B[1, 0]
    c, d = payoff_B[0, 1], payoff_B[1, 1]
    denom_p = (a - b) - (c - d)
    if abs(denom_p) < 1e-9:
        return None
    p = (d - b) / denom_p

    # A's indifference: row 0 vs row 1 given B plays col 0 with prob q
    a2, b2 = payoff_A[0, 0], payoff_A[0, 1]
    c2, d2 = payoff_A[1, 0], payoff_A[1, 1]
    denom_q = (a2 - b2) - (c2 - d2)
    if abs(denom_q) < 1e-9:
        return None
    q = (d2 - b2) / denom_q

    if 0 <= p <= 1 and 0 <= q <= 1:
        return (p, q)
    return None

# 3)Best response dynamics

def best_response_dynamics(payoff_A, payoff_B, start=(0, 0), max_iter=50):
    """
    Simulate alternating best-response play from a starting profile.
    Each step: one player best-responds to the other's current strategy.

    Returns (trajectory, converged).
    """
    i, j = start
    trajectory = [(i, j)]
    for _ in range(max_iter):
        i_new = int(np.argmax(payoff_A[:, j]))
        j_new = int(np.argmax(payoff_B[i_new, :]))
        if (i_new, j_new) == (i, j):
            return trajectory, True
        i, j = i_new, j_new
        trajectory.append((i, j))
    return trajectory, False

# 4)Tests on classic games

games = {
    "Prisoner's Dilemma": (
        np.array([[-1, -3], [0, -2]]),
        np.array([[-1, 0], [-3, -2]]),
        ["Cooperate", "Defect"],
    ),
    "Battle of the Sexes": (
        np.array([[3, 0], [0, 2]]),
        np.array([[2, 0], [0, 3]]),
        ["Opera", "Fight"],
    ),
    "Matching Pennies": (
        np.array([[1, -1], [-1, 1]]),
        np.array([[-1, 1], [1, -1]]),
        ["Heads", "Tails"],
    ),
}

print("Pure-Strategy Nash Equilibrium Finder")
print("=" * 55)
for name, (pA, pB, strats) in games.items():
    eq = find_pure_nash(pA, pB)
    eq_named = [(strats[i], strats[j]) for i, j in eq]
    mixed    = find_mixed_nash_2x2(pA, pB)
    traj, conv = best_response_dynamics(pA, pB)
    print(f"\n{name}:")
    print(f"  Pure NE : {eq_named if eq_named else 'None'}")
    if mixed:
        print(f"  Mixed NE: p(row 0)={mixed[0]:.4f}, q(col 0)={mixed[1]:.4f}")
    print(f"  BRD from (0,0): {traj} | converged={conv}")

# 5)Bertrand duopoly

print("\n" + "=" * 55)
print("Bertrand Duopoly — Price Competition")
print("=" * 55)

COST        = 10
MARKET_SIZE = 100
price_grid  = np.arange(COST, COST + 30, 1)
n_prices    = len(price_grid)

def bertrand_profit(p_self, p_other, cost, demand_intercept):
    if p_self < p_other:
        return (p_self - cost) * (demand_intercept - p_self)
    elif p_self > p_other:
        return 0.0
    else:
        return (p_self - cost) * (demand_intercept - p_self) / 2.0

bA = np.zeros((n_prices, n_prices))
bB = np.zeros((n_prices, n_prices))
for i, p1 in enumerate(price_grid):
    for j, p2 in enumerate(price_grid):
        bA[i, j] = bertrand_profit(p1, p2, COST, MARKET_SIZE)
        bB[i, j] = bertrand_profit(p2, p1, COST, MARKET_SIZE)

eq_b = find_pure_nash(bA, bB)
eq_prices = [(price_grid[i], price_grid[j]) for i, j in eq_b]
print(f"Marginal cost : {COST}")
print(f"NE price pairs: {eq_prices[:4]}")
print("=> Bertrand paradox: price → marginal cost, zero profit, 2 firms")

# 6)Cournot duopoly: reaction functions & NE

print("\n" + "=" * 55)
print("Cournot Duopoly — Quantity Competition")
print("=" * 55)

a_d, b_d, c_d = 100, 1, 10   # demand: P = a - b*(q1+q2), cost c per unit

def cournot_reaction(q_other, a, b, c):
    #Best response output for one firm given rival's output
    return max(0.0, (a - c) / (2 * b) - q_other / 2)

def cournot_ne_symmetric(a, b, c):
    #Symmetric Cournot NE analytical solution
    q_ne = (a - c) / (3 * b)
    P_ne = a - b * (2 * q_ne)
    pi_ne = (P_ne - c) * q_ne
    return q_ne, P_ne, pi_ne

def monopoly_benchmark(a, b, c):
    #Joint profit maximising monopoly output
    Q_m  = (a - c) / (2 * b)
    P_m  = a - b * Q_m
    pi_m = (P_m - c) * Q_m   # total
    return Q_m / 2, P_m, pi_m / 2   # per firm split

q_ne, P_ne, pi_ne = cournot_ne_symmetric(a_d, b_d, c_d)
q_m, P_m, pi_m   = monopoly_benchmark(a_d, b_d, c_d)
q_comp = (a_d - c_d) / (2 * b_d)   # perfectly competitive per firm

print(f"\n{'Scenario':<28} {'q*':>8} {'P*':>8} {'Profit*':>10}")
print("-" * 56)
print(f"{'Cournot NE (2 firms)':<28} {q_ne:>8.2f} {P_ne:>8.2f} {pi_ne:>10.2f}")
print(f"{'Monopoly/Cartel (per firm)':<28} {q_m:>8.2f} {P_m:>8.2f} {pi_m:>10.2f}")
print(f"{'Competitive (per firm)':<28} {q_comp:>8.2f} {c_d:>8.2f} {'0':>10}")
print(f"\nCournot NE lies between monopoly (q={q_m:.1f}) and competitive (q={q_comp:.1f})")

# OPEC style: 13 firms
print("\nCournot NE for n firms (OPEC scenario, a=100, b=1, c=10):")
print(f"{'n':>5} {'q_i':>8} {'Q_total':>10} {'Price':>8} {'Profit_i':>10}")
print("-" * 45)
for n in [2, 5, 8, 13, 20]:
    q_i  = (a_d - c_d) / ((n + 1) * b_d)
    Q    = n * q_i
    P    = a_d - b_d * Q
    pi_i = (P - c_d) * q_i
    print(f"{n:>5} {q_i:>8.2f} {Q:>10.2f} {P:>8.2f} {pi_i:>10.2f}")


# 7)Figures

q_range = np.linspace(0, (a_d - c_d) / (2 * b_d), 200)
react   = np.array([cournot_reaction(q, a_d, b_d, c_d) for q in q_range])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Cournot reaction functions
axes[0].plot(react,   q_range, color="steelblue",  lw=2, label="Firm 1: $q_1^*(q_2)$")
axes[0].plot(q_range, react,   color="darkorange",  lw=2, label="Firm 2: $q_2^*(q_1)$")
axes[0].scatter([q_ne], [q_ne], color="red", s=100, zorder=5,
                label=f"Cournot NE: ({q_ne:.1f}, {q_ne:.1f})")
axes[0].scatter([q_m],  [q_m],  color="gray", s=70, marker="s", zorder=5,
                label=f"Monopoly: ({q_m:.1f}, {q_m:.1f})")
axes[0].annotate("Monopoly\n(cartel)", xy=(q_m, q_m), xytext=(28, 8),
                 arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9, color="gray")
axes[0].set_xlabel("Firm 1 output $q_1$"); axes[0].set_ylabel("Firm 2 output $q_2$")
axes[0].set_title("Cournot Duopoly: Reaction Functions")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
axes[0].set_xlim(0, 55); axes[0].set_ylim(0, 55)

# Bertrand profit landscape
im = axes[1].imshow(bA, origin="lower",
                    extent=[price_grid[0], price_grid[-1],
                            price_grid[0], price_grid[-1]],
                    cmap="viridis", aspect="auto")
plt.colorbar(im, ax=axes[1], label="Firm A profit")
axes[1].scatter([price_grid[j] for _, j in eq_b],
                [price_grid[i] for i, _ in eq_b],
                color="red", marker="x", s=80, zorder=5, label="Nash Equilibria")
axes[1].legend(); axes[1].set_xlabel("Firm B price"); axes[1].set_ylabel("Firm A price")
axes[1].set_title("Bertrand: Profit Landscape (NE at MC)")
plt.tight_layout()
plt.savefig("figures/w2_competition.png", dpi=150)
print("\nSaved: figures/w2_competition.png")
