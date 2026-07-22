"""
Week 1: Foundations of Game Theory
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1)Payoff matrix representation

strategies = ["Cooperate", "Defect"]

#Entry (i,j): Player A's payoff when A plays row i, B plays col j
payoff_A = np.array([
    [-1, -3],   # A Cooperates: B Coop=-1, B Defect=-3
    [ 0, -2],   # A Defects:    B Coop= 0, B Defect=-2
])

payoff_B = np.array([
    [-1,  0],
    [-3, -2],
])

print("=" * 55)
print("Prisoner's Dilemma — Normal-Form Representation")
print("=" * 55)
print(f"\n{'':15}{'B: Cooperate':20}{'B: Defect'}")
for i, label in enumerate(["A: Cooperate", "A: Defect"]):
    print(f"{label:15}  ({payoff_A[i,0]:>2},{payoff_B[i,0]:>2})              "
          f"({payoff_A[i,1]:>2},{payoff_B[i,1]:>2})")

# 2)Strict dominance check


def strictly_dominates(payoff_matrix, row_dominating, row_dominated):
    """
    Check if row_dominating strictly dominates row_dominated for the row player.
    Dominance: payoff is strictly higher in every column.
    """
    return np.all(payoff_matrix[row_dominating, :] > payoff_matrix[row_dominated, :])

print("\nDominance Analysis")
print("-" * 55)
for name, mat in [("A", payoff_A), ("B", payoff_B.T)]:
    dom = strictly_dominates(mat, 1, 0)   # Defect (row 1) vs Cooperate (row 0)
    print(f"Player {name}: Defect strictly dominates Cooperate? {dom}")

print("\nConclusion:")
print("  => Nash Equilibrium: (Defect, Defect)")
print("  => Social optimum:   (Cooperate, Cooperate)")
print("  => Gap: NE yields (-2,-2) vs optimum (-1,-1)")

# 3)Social welfare analysis

social_welfare = payoff_A + payoff_B
print("\nSocial Welfare (A payoff + B payoff):")
for i, row in enumerate(strategies):
    for j, col in enumerate(strategies):
        print(f"  ({row}, {col}): {social_welfare[i,j]}")

ne_welfare  = social_welfare[1,1]   # (Defect, Defect)
opt_welfare = social_welfare[0,0]   # (Cooperate, Cooperate)
print(f"\nWelfare at NE:       {ne_welfare}")
print(f"Welfare at optimum:  {opt_welfare}")
print(f"Efficiency loss:     {opt_welfare - ne_welfare} units")

# 4)Financial application: Cournot (OPEC-style) symmetric equilibrium preview


print("\n" + "=" * 55)
print("Financial Context: OPEC as Prisoner's Dilemma")
print("=" * 55)

a, b, c = 100, 1, 10   # demand intercept, slope, marginal cost

def cournot_ne(n_firms, a, b, c):
    #Symmetric Cournot Nash equilibrium output and price
    q_star   = (a - c) / ((n + 1) * b)
    Q_star   = n * q_star
    P_star   = a - b * Q_star
    profit   = (P_star - c) * q_star
    return q_star, Q_star, P_star, profit

def monopoly_output(a, b, c):
    #Monopoly (cartel) outcome
    Q_m    = (a - c) / (2 * b)
    P_m    = a - b * Q_m
    profit = (P_m - c) * Q_m / 1   # per firm if split equally
    return Q_m, P_m, profit

Q_m, P_m, pi_m = monopoly_output(a, b, c)
print(f"\nMonopoly (OPEC cartel fully enforced):")
print(f"  Total output: {Q_m:.2f}, Price: {P_m:.2f}, Profit per firm: {pi_m:.2f}")

for n in [2, 5, 13]:   # 13 ≈ number of OPEC members
    q, Q, P, pi = cournot_ne(n, a, b, c)
    print(f"\nCournot NE ({n} firms, OPEC-like):")
    print(f"  Each firm: q={q:.2f}, Total Q={Q:.2f}, Price={P:.2f}, Profit={pi:.2f}")
    print(f"  Cartel gain per firm: {pi_m - pi:.2f} (if they could enforce it)")

# 5. Visualisation

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, mat, title in zip(
        axes, [payoff_A, payoff_B, social_welfare],
        ["Player A Payoff", "Player B Payoff", "Social Welfare (A+B)"]):
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-4, vmax=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(strategies)
    ax.set_yticks([0, 1]); ax.set_yticklabels(strategies)
    ax.set_title(title, fontsize=12)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, mat[i, j], ha="center", va="center",
                    fontsize=14, fontweight="bold")
    ax.set_xlabel("Player B"); ax.set_ylabel("Player A")
plt.tight_layout()
plt.savefig("figures/w1_pd.png", dpi=150)
print("\nSaved: figures/w1_pd.png")
