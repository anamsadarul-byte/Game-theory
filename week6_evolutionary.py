"""
Week 6: Evolutionary Game Theory
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

np.random.seed(42)

# 1)Replicator dynamics engine

def replicator_dynamics(payoff_matrix, x0, T=200.0, dt=0.05):
    """
    Simulate the replicator dynamic for an n strategy symmetric game.

    The replicator equation:
        dx_i/dt = x_i * (f_i(x) - f_bar(x))
    where:
        f_i(x) = sum_j M_{ij} * x_j      (strategy i's fitness)
        f_bar(x) = x^T * f(x)            (mean population fitness)

    Parameters
    payoff_matrix : ndarray (n, n)
        M[i,j] = payoff to strategy i against strategy j
    x0 : array-like (n,)
        Initial population frequencies (must sum to 1)
    T  : float: total simulation time
    dt : float: Euler integration step size

    Returns
    ts : ndarray (steps+1,) : time points
    xs : ndarray (steps+1, n): frequency trajectories
    """
    x  = np.array(x0, dtype=float)
    xs = [x.copy()]
    ts = [0.0]
    t  = 0.0

    for _ in range(int(T / dt)):
        f_vec   = payoff_matrix @ x      # vector of strategy fitnesses
        f_bar   = x @ f_vec              # mean fitness
        dx      = x * (f_vec - f_bar)   # replicator equation
        x       = np.clip(x + dt * dx, 0.0, 1.0)
        x      /= x.sum()               # renormalise for numerical stability
        t      += dt
        xs.append(x.copy())
        ts.append(t)

    return np.array(ts), np.array(xs)


def ess_check(payoff_matrix, sigma_star, epsilon=1e-6):
    """
    Check if sigma_star is an ESS by verifying both ESS conditions
    for all pure strategies as potential mutants
    sigma_star: array of mixed-strategy frequencies
    Returns list of (mutant, condition_1, condition_2, is_ESS_against)
    """
    n = len(sigma_star)
    u_star_vs_star = sigma_star @ payoff_matrix @ sigma_star
    results = []
    for i in range(n):
        e_i = np.zeros(n); e_i[i] = 1.0
        u_mutant_vs_star = e_i @ payoff_matrix @ sigma_star
        cond1 = u_star_vs_star > u_mutant_vs_star + epsilon
        # If condition 1 fails (tie), check condition 2
        u_star_vs_mutant  = sigma_star @ payoff_matrix @ e_i
        u_mutant_vs_mutant = e_i @ payoff_matrix @ e_i
        cond2 = u_star_vs_mutant > u_mutant_vs_mutant + epsilon
        is_ess = cond1 or (not cond1 and cond2)
        results.append((i, cond1, cond2, is_ess))
    return results

# 2)Hawk Dove game

V, C = 4, 6   # resource value, fighting cost

hawk_dove_matrix = np.array([
    [(V - C) / 2,  V   ],   # Hawk vs (Hawk, Dove)
    [0,            V / 2],   # Dove vs (Hawk, Dove)
])

# Analytical ESS: p* where Hawk and Dove have equal expected fitness
# Hawk fitness: p*(V-C)/2 + (1-p*)*V = p* (V-C)/2 + V - p*V
# Dove fitness: p**0 + (1-p*)*V/2
# Setting equal: p* = V/C
ess_hawk = V / C

print("Hawk-Dove Game")
print("=" * 50)
print(f"V = {V}, C = {C}")
print(f"Payoff matrix:")
print(f"  H vs H: {(V-C)/2:.1f}    H vs D: {V:.1f}")
print(f"  D vs H: {0:.1f}    D vs D: {V/2:.1f}")
print(f"\nAnalytical ESS: p* = V/C = {V}/{C} = {ess_hawk:.4f}")

# ESS verification
sigma_ess = np.array([ess_hawk, 1 - ess_hawk])
ess_results = ess_check(hawk_dove_matrix, sigma_ess)
print(f"\nESS verification at p*={ess_hawk:.3f}:")
strategy_names = ["Hawk", "Dove"]
for i, c1, c2, is_ess in ess_results:
    print(f"  vs {strategy_names[i]} mutant: cond1={c1}, cond2={c2} => ESS={is_ess}")

# Simulate replicator dynamics from two starting points
ts1, xs1 = replicator_dynamics(hawk_dove_matrix, x0=[0.9, 0.1])
ts2, xs2 = replicator_dynamics(hawk_dove_matrix, x0=[0.1, 0.9])
print(f"\nReplicator dynamics convergence:")
print(f"  Start (0.9 Hawk): final Hawk freq = {xs1[-1,0]:.5f}")
print(f"  Start (0.1 Hawk): final Hawk freq = {xs2[-1,0]:.5f}")
print(f"  ESS target:                         {ess_hawk:.5f}")

# 3)Prisoner's Dilemma under replicator dynamics

b, c_pd = 3, 1   # benefit of mutual cooperation,cost to cooperator

pd_matrix = np.array([
    [b - c_pd, -c_pd],   # Cooperate vs (C, D)
    [b,         0   ],   # Defect    vs (C, D)
])

print("\n" + "=" * 50)
print("Prisoner's Dilemma: Evolutionary Dynamics")
print("=" * 50)
print(f"b={b}, c={c_pd}")
print(f"Payoff matrix:")
print(f"  CC={b-c_pd:.0f}  CD={-c_pd:.0f}")
print(f"  DC={b:.0f}   DD={0:.0f}")
print("Defect strictly dominates => cooperation goes extinct")

ts_pd1, xs_pd1 = replicator_dynamics(pd_matrix, x0=[0.9, 0.1])
ts_pd2, xs_pd2 = replicator_dynamics(pd_matrix, x0=[0.1, 0.9])
print(f"Final cooperator freq (start=0.9): {xs_pd1[-1,0]:.6f}")
print(f"Final cooperator freq (start=0.1): {xs_pd2[-1,0]:.6f}")

# 4)Agent based PD on a scale-free network (network reciprocity)

def agent_based_pd(n_agents=100, n_rounds=200, m_edges=3,
                   noise=0.02, b=3, c=1):
    """
    Agents play pairwise PD on a Barabási Albert scale free network

    Update rule: each agent imitates the best performing neighbour
    (inclusive of themselves), with `noise` probability of random mutation

    Parameters
    n_agents : int: population size
    n_rounds : int: number of time steps
    m_edges  : int: edges added per new node in BA graph
    noise    : float: mutation probability per step
    b, c     : float: PD benefit and cost parameters

    Returns
    coop_history : list of float fraction cooperating per round
    """
    payoff_matrix = np.array([[b - c, -c], [b, 0]])
    G = nx.barabasi_albert_graph(n_agents, m_edges, seed=42)

    # Initial strategies: 0=Defect, 1=Cooperate (50/50 split)
    strategies = np.random.choice([0, 1], size=n_agents, p=[0.5, 0.5])
    coop_history = [strategies.mean()]

    for _ in range(n_rounds):
        # Compute accumulated payoffs from pairwise edge interactions
        scores = np.zeros(n_agents)
        for u, v in G.edges():
            scores[u] += payoff_matrix[strategies[u], strategies[v]]
            scores[v] += payoff_matrix[strategies[v], strategies[u]]

        # Imitation update with mutation
        new_strategies = strategies.copy()
        for i in G.nodes():
            neighbours = list(G.neighbors(i))
            if neighbours:
                candidates = neighbours + [i]
                best_nbr   = max(candidates, key=lambda k: scores[k])
                if np.random.rand() < noise:
                    new_strategies[i] = 1 - strategies[i]   # random mutation
                else:
                    new_strategies[i] = strategies[best_nbr]  # imitate best

        strategies = new_strategies
        coop_history.append(strategies.mean())

    return coop_history

print("\n" + "=" * 50)
print("Agent-Based PD on Barabási-Albert Network")
print("=" * 50)
coop_hist = agent_based_pd(n_agents=100, n_rounds=200)
steady = np.mean(coop_hist[50:])
print(f"Steady state cooperation (rounds 50-200): {steady:.4f}")
print(f"Contrast with well mixed result:          {xs_pd1[-1,0]:.4f}")
print(f"Network reciprocity boost: +{(steady - xs_pd1[-1,0])*100:.1f} percentage points")


# 5)Trading strategy evolution

print("\n" + "=" * 50)
print("Financial Application: Trading Strategy Evolution")
print("=" * 50)

# Three strategy types compete in a stylised financial market:
# T = Trend follower (momentum): profits when prices trend
# M = Mean reverter: profits when prices revert
# C = Contrarian: fades extremes, bets against consensus
# Payoff matrix (row beats column):
# T exploits M (M fades trend T profits from); M exploits C (C goes wrong way);
# C exploits T (C fades trend extrapolation by T)
# This cyclic structure produces a mixed ESS, not a pure one

trading_matrix = np.array([
    #  T      M      C
    [ 0.00,  0.50, -0.30],   # T vs (T, M, C)
    [-0.50,  0.00,  0.40],   # M vs (T, M, C)
    [ 0.30, -0.40,  0.00],   # C vs (T, M, C)
])

strategy_names = ["Trend follower", "Mean reverter", "Contrarian"]

# Simulate from two starting points
ts_tr1, xs_tr1 = replicator_dynamics(trading_matrix, x0=[0.70, 0.20, 0.10])
ts_tr2, xs_tr2 = replicator_dynamics(trading_matrix, x0=[1/3, 1/3, 1/3])

print("Starting from 70% Trend followers (bull market initial condition):")
for i, name in enumerate(strategy_names):
    print(f"  Final {name} frequency: {xs_tr1[-1,i]:.4f}")

print("\nStarting from equal 1/3 mix:")
for i, name in enumerate(strategy_names):
    print(f"  Final {name} frequency: {xs_tr2[-1,i]:.4f}")

print("\nObservations:")
print("  Trend followers initially dominant, but attract mean reverters")
print("  Cyclic dynamics: no single strategy permanently wins")
print("  Consistent with empirical decay of momentum strategy alpha over time")
print("  As T followers dominate, market becomes predictable => M/C strategies profitable")

# 6. Figures

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
colors_t = ["darkorange", "steelblue", "green"]

# Top left: Hawk Dove
axes[0,0].plot(ts1, xs1[:,0], color="firebrick", lw=2, label="Hawk (start=0.9)")
axes[0,0].plot(ts2, xs2[:,0], color="firebrick", lw=2, linestyle="-",
               label="Hawk (start=0.1)")
axes[0,0].axhline(ess_hawk, color="black", linestyle=":", lw=1.5,
                  label=f"ESS: p*={ess_hawk:.3f}")
axes[0,0].set_xlabel("Time"); axes[0,0].set_ylabel("Hawk frequency")
axes[0,0].set_title("Hawk Dove: Convergence to ESS")
axes[0,0].legend(fontsize=9); axes[0,0].grid(alpha=0.3); axes[0,0].set_ylim(0,1)

# Top right: PD well mixed
axes[0,1].plot(ts_pd1, xs_pd1[:,0], color="steelblue", lw=2,
               label="Cooperators (start=0.9)")
axes[0,1].plot(ts_pd2, xs_pd2[:,0], color="steelblue", lw=2, linestyle="-",
               label="Cooperators (start=0.1)")
axes[0,1].axhline(0, color="black", linestyle=":", lw=1.5, label="All Defect")
axes[0,1].set_xlabel("Time"); axes[0,1].set_ylabel("Cooperator frequency")
axes[0,1].set_title("PD Well-Mixed: Defectors Win")
axes[0,1].legend(fontsize=9); axes[0,1].grid(alpha=0.3); axes[0,1].set_ylim(-0.05, 1.05)

# Bottom left: Trading strategy evolution
for i, (name, col) in enumerate(zip(strategy_names, colors_t)):
    axes[1,0].plot(ts_tr1, xs_tr1[:,i], color=col, lw=2, label=name)
axes[1,0].set_xlabel("Time"); axes[1,0].set_ylabel("Strategy frequency")
axes[1,0].set_title("Trading Strategy Evolution\n(Start: 70% Trend followers)")
axes[1,0].legend(fontsize=9); axes[1,0].grid(alpha=0.3); axes[1,0].set_ylim(0,1)

# Bottom right: Network cooperation
axes[1,1].plot(coop_hist, color="steelblue", lw=1.5)
axes[1,1].axhline(steady, color="gray", linestyle="-",
                  label=f"Steady state: {steady:.2f}")
axes[1,1].set_xlabel("Round"); axes[1,1].set_ylabel("Fraction cooperating")
axes[1,1].set_title("Agent Based PD on Scale-Free Network\n(Network reciprocity sustains cooperation)")
axes[1,1].legend(fontsize=9); axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/w6_evolutionary.png", dpi=150)
print("\nSaved: figures/w6_evolutionary.png")
