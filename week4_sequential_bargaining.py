"""
Week 4: Sequential Games and Negotiation
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1)Market-entry deterrence game: backward induction

print("Market Entry Deterrence Game: Backward Induction")
print("=" * 55)

# Game tree:
#                     Entrant
#                    /        \
#               Enter          Stay Out
#                 |                |
#            Incumbent          (0, +5)
#            /        \
#         Fight      Accommodate
#       (-2, -1)      (+2, +2)

PAYOFFS = {
    "Enter -> Fight":       (-2, -1),   # (Entrant, Incumbent)
    "Enter -> Accommodate": (+2, +2),
    "Stay Out":           ( 0, +5),
}

print("\nTerminal payoffs (Entrant, Incumbent):")
for outcome, (e, inc) in PAYOFFS.items():
    print(f"  {outcome:<26}: Entrant={e:+d}, Incumbent={inc:+d}")

# Step 1) Solve Incumbent's subgame (after Entry)
fight_u_inc = PAYOFFS["Enter -> Fight"][1]
accom_u_inc = PAYOFFS["Enter -> Accommodate"][1]
incumbent_choice = "Accommodate" if accom_u_inc > fight_u_inc else "Fight"
incumbent_u      = max(fight_u_inc, accom_u_inc)

print(f"\nStep 1: Incumbent's node (entry has occurred):")
print(f"  Fight payoff = {fight_u_inc}, Accommodate payoff = {accom_u_inc}")
print(f"  => Incumbent chooses: {incumbent_choice}")

# Step 2) Solve Entrant's decision anticipating Incumbent's choice
entrant_if_enter = PAYOFFS[f"Enter -> {incumbent_choice}"][0]
entrant_if_out   = PAYOFFS["Stay Out"][0]
entrant_choice   = "Enter" if entrant_if_enter > entrant_if_out else "Stay Out"

print(f"\nStep 2 — Entrant's node:")
print(f"  Enter -> {incumbent_choice} payoff = {entrant_if_enter}, Stay Out = {entrant_if_out}")
print(f"  => Entrant chooses: {entrant_choice}")

final_payoff = PAYOFFS[f"Enter -> {incumbent_choice}"]
print(f"\nSubgame Perfect Equilibrium:")
print(f"  Entrant: {entrant_choice} Incumbent: {incumbent_choice} (if entry occurs)")
print(f"  Equilibrium payoffs: Entrant={final_payoff[0]:+d}, Incumbent={final_payoff[1]:+d}")
print(f"\n  Key insight: Incumbent's 'Fight' threat is NOT credible.")
print(f"  Once entry occurs, Fighting costs -{abs(fight_u_inc)} vs. Accommodating +{accom_u_inc}.")
print(f"  Without a binding commitment (capacity, reputation), the threat is empty")


# 2) Generic backward induction on a game tree (dict representation)

def backward_induction(tree, node="root"):
    """
    Generic backward induction solver for finite perfect-information games.

    tree: dict of dicts
      Each node is either:
         A dict with key "player" (int), "children" (dict: action -> node_name)
         A dict with key "payoffs" (tuple) indicating a terminal node

    Returns (best_action_dict, value_dict)
    """
    node_data = tree[node]

    # Terminal node
    if "payoffs" in node_data:
        return {}, node_data["payoffs"]

    player   = node_data["player"]
    children = node_data["children"]

    best_action = None
    best_payoffs = None
    action_map = {}

    for action, child_node in children.items():
        sub_actions, sub_payoffs = backward_induction(tree, child_node)
        action_map[action] = (sub_actions, sub_payoffs)
        if best_payoffs is None or sub_payoffs[player] > best_payoffs[player]:
            best_payoffs = sub_payoffs
            best_action  = action

    result_actions = {node: best_action}
    result_actions.update(action_map[best_action][0])
    return result_actions, best_payoffs


# Build the market entry game tree
entry_tree = {
    "root": {"player": 0, "children": {"Enter": "incumbent_node", "Stay Out": "stay_out"}},
    "incumbent_node": {"player": 1, "children": {"Fight": "fight_terminal", "Accommodate": "accom_terminal"}},
    "stay_out":        {"payoffs": (0, 5)},
    "fight_terminal":  {"payoffs": (-2, -1)},
    "accom_terminal":  {"payoffs": (2, 2)},
}

spe_actions, spe_payoffs = backward_induction(entry_tree)
print(f"\nGeneric solver output:")
print(f"  SPE actions : {spe_actions}")
print(f"  SPE payoffs : {spe_payoffs}")

# 3. Rubinstein alternating-offers bargaining

print("\n" + "=" * 55)
print("Rubinstein Alternating Offers Bargaining")
print("=" * 55)

def rubinstein_finite(delta, rounds):
    """
    Finite horizon Rubinstein backward induction
    Splits a pie of size 1 over `rounds` rounds, symmetric discount delta

    Recursion (from terminal round backwards):
        v[1] = 1              (proposer in final round takes everything)
        v[k] = 1 - delta * v[k-1]   (proposer offers responder just enough
                                      to be indifferent vs waiting one round)
    Returns first proposer's equilibrium share
    """
    v = 1.0
    for _ in range(2, rounds + 1):
        v = 1.0 - delta * v
    return v


def rubinstein_infinite(delta):
    """
    Closed form infinite horizon solution: v* = 1/(1+delta)
    Derived by solving the fixed point v = 1 - delta*v => v = 1/(1+delta)
    """
    return 1.0 / (1.0 + delta)


def rubinstein_asymmetric(delta_1, delta_2):
    """
    Asymmetric discount rates
    Infinite-horizon SPE: Player 1 (proposer) gets:
        v1* = (1 - delta_2) / (1 - delta_1 * delta_2)
    """
    v1 = (1 - delta_2) / (1 - delta_1 * delta_2)
    v2 = 1 - v1
    return v1, v2


print(f"\n{'delta':>6} {'Finite (200 rds)':>18} {'Infinite 1/(1+d)':>18} {'Responder share':>16} {'Gap':>10}")
print("" * 72)
test_deltas = [0.10, 0.30, 0.50, 0.70, 0.90, 0.95, 0.99]
for d in test_deltas:
    fin = rubinstein_finite(d, 200)
    inf = rubinstein_infinite(d)
    print(f"{d:>6.2f} {fin:>18.5f} {inf:>18.5f} {1-inf:>16.5f} {abs(fin-inf):>10.5f}")

print("\nAsymmetric discount rates (Player 1 more patient):")
print(f"{'delta_1':>8} {'delta_2':>8} {'v1*':>10} {'v2*':>10} {'1\'s advantage':>14}")
print("" * 55)
for d1, d2 in [(0.9, 0.5), (0.9, 0.7), (0.9, 0.9), (0.5, 0.9)]:
    v1, v2 = rubinstein_asymmetric(d1, d2)
    print(f"{d1:>8.2f} {d2:>8.2f} {v1:>10.4f} {v2:>10.4f} {v1-0.5:>+14.4f}")

print("\nKey results:")
print("  1. As delta -> 1: v* -> 0.5 (patience equalises bargaining power)")
print("  2. More patient player gets a larger share (v1*>0.5 when delta_1 > delta_2)")
print("  3. First-mover advantage: v* > 0.5 for any finite delta")
print("  4. Convergence rate of finite -> infinite is delta^T (slow when delta~1)")

# 4. Financial application: M&A takeover game

print("\n" + "=" * 55)
print("Financial Application: M&A Takeover Bargaining")
print("=" * 55)

# Acquirer (A) and Target (T) negotiate over synergy value V_s
# A's discount rate r_A (quarterly), T's discount rate r_T
# Rubinstein delta_i = 1/(1+r_i) for each party

V_synergy = 500e6   # $500M synergy value to be split

scenarios = [
    ("Equal negotiating power",        0.02, 0.02),
    ("Acquirer more patient (lower r)", 0.01, 0.03),
    ("Target more patient (lower r)",   0.03, 0.01),
    ("Acquirer desperate (high r)",     0.05, 0.01),
]

print(f"\nSynergy value: ${V_synergy/1e6:.0f}M")
print(f"\n{'Scenario':<35} {'A share':>9} {'T gets':>12} {'A gets':>12}")
print("-" * 72)
for scenario, r_A, r_T in scenarios:
    d_A = 1 / (1 + r_A)
    d_T = 1 / (1 + r_T)
    v_A, v_T = rubinstein_asymmetric(d_A, d_T)
    print(f"{scenario:<35} {v_A:>8.1%}  ${v_T*V_synergy/1e6:>10.1f}M  ${v_A*V_synergy/1e6:>10.1f}M")

print("\nImplication: Acquirer's cost of capital (r_A) directly determines the")
print("  takeover premium: firms with cheap capital extract more of the synergy.")

# 5. Figures

deltas_fine = np.linspace(0.01, 0.99, 300)
inf_shares  = rubinstein_infinite(deltas_fine)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: bargaining shares vs delta
axes[0].plot(deltas_fine, inf_shares,   color="darkgreen", lw=2,
             label="Proposer: $1/(1+\\delta)$")
axes[0].plot(deltas_fine, 1-inf_shares, color="darkgreen", lw=2, linestyle="-",
             label="Responder: $\\delta/(1+\\delta)$")
axes[0].axhline(0.5, color="gray", linestyle=":", lw=1.5, label="Equal split (0.5)")
axes[0].fill_between(deltas_fine, inf_shares, 0.5,
                     where=(inf_shares > 0.5), alpha=0.1, color="green",
                     label="Proposer advantage")
axes[0].scatter(test_deltas, [rubinstein_infinite(d) for d in test_deltas],
                color="darkgreen", s=60, zorder=5)
axes[0].set_xlabel("Discount factor $\\delta$")
axes[0].set_ylabel("Equilibrium share")
axes[0].set_title("Rubinstein Bargaining: Patience vs. Power")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

# Right: M&A scenario comparison
labels = [s[0].replace(" ", "\n") for s in scenarios]
a_shares = []
for _, r_A, r_T in scenarios:
    d_A, d_T = 1/(1+r_A), 1/(1+r_T)
    v_A, _ = rubinstein_asymmetric(d_A, d_T)
    a_shares.append(v_A * V_synergy / 1e6)

t_shares = [V_synergy/1e6 - a for a in a_shares]
x = np.arange(len(scenarios))
w = 0.35
axes[1].bar(x - w/2, a_shares, width=w, label="Acquirer gets ($M)", color="steelblue")
axes[1].bar(x + w/2, t_shares, width=w, label="Target gets ($M)",   color="darkorange")
axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=7.5)
axes[1].set_ylabel("Share of $500M synergy ($M)")
axes[1].set_title("M&A Rubinstein: Synergy Split by Negotiating Position")
axes[1].legend(fontsize=9); axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("figures/w4_bargaining.png", dpi=150)
print("\nSaved: figures/w4_bargaining.png")
