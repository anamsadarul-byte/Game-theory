```bash
pip install -r requirements.txt
python run_all.py
```

Or run individual weeks:

```bash
python week1_foundations.py
python week2_nash_competition.py
# ... etc
```

All figures are written to `./figures/`.

## File Structure

```
sos_gametheory/

 week1_foundations.py            Payoff matrices, strict dominance, PD
 week2_nash_competition.py       Pure/mixed NE, BRD, Bertrand, Cournot
 week3_auctions.py               1st/2nd-price, Revenue Equivalence, real auctions
 week4_sequential_bargaining.py  Backward induction, Rubinstein, M&A model
 week5_congestion_poa.py         Pigou PoA, Braess paradox, HFT arms race
 week6_evolutionary.py           Replicator dynamics, ESS, network reciprocity
 week7_multiagent_kyle.py        Repeated auctions, minimax, Blotto, Kyle model
 week8_synthesis.py              Cross-game PoA, financial market efficiency

 run_all.py                      Run full pipeline in one command
 requirements.txt
 figures/                        All output figures
```

## What Each Week do?

# Week 1: Foundations
- Normal-form game payoff matrix (NumPy)
- Strict dominance check for any $(m \times n)$ game
- Social welfare analysis of Prisoner's Dilemma
- Cournot NE preview: OPEC-style $n$-firm output table

# Week 2: Nash Equilibrium
- **`find_pure_nash(pA, pB)`** brute-force pure NE finder for any $(m \times n)$ game
- **`find_mixed_nash_2x2(pA, pB)`** mixed NE via indifference conditions
- **`best_response_dynamics(pA, pB, start)`** convergence simulation
- Bertrand duopoly over a $30 \times 30$ price grid
- Cournot reaction functions: analytic NE + numerical verification
- OPEC table: NE output/price/profit for $n = 2, 5, 8, 13, 20$ firms

# Week 3: Auctions
- **`simulate_second_price(n, T)`** Vickrey auction Monte Carlo
- **`simulate_first_price(n, T)`** BNE shading $b(v) = v(n-1)/n$
- **`analytic_revenue(n)`** closed form $H(n-1)/(n+1)$
- Revenue Equivalence verification: $\Delta\mu < \$0.01$ over 10,000 trials
- Bid-shading table: shade factor and expected revenue vs. $n = 2 \ldots 15$
- Real-world benchmarks: FCC AWS-3 (\$41.33B), Google IPO, US Treasury

# Week 4: Sequential Games & Bargaining
- **`backward_induction(tree, node)`** generic recursive BI solver
- Market-entry deterrence game (SPE derivation)
- **`rubinstein_finite(delta, rounds)`** backward induction recursion
- **`rubinstein_asymmetric(d1, d2)`** asymmetric discount rates
- M&A model: acquirer/target synergy split vs. cost of capital

# Week 5: Congestion
- Pigou social cost function + analytic PoA = 4/3 verification
- Braess paradox: before/after adding zero-cost edge
- **HFT latency arms race**: symmetric NE investment, social optimum, waste
- Key market facts: HFT volume share, co-location costs, microwave latency

# Week 6: Evolutionary Game Theory
- **`replicator_dynamics(M, x0, T, dt)`** Euler integration of replicator ODE
- **`ess_check(M, sigma_star)`** ESS verification (both conditions)
- Hawk-Dove: ESS at $p^* = V/C$, convergence from two starting points
- PD: cooperators go extinct in well mixed population
- **`agent_based_pd(n, rounds, m, noise)`** BA network + imitation update
- Trading strategy evolution: trend-follower / mean-reverter / contrarian

# Week 7: Multi-Agent & Kyle Model
- **`repeated_auction_learning(n, rounds, lr)`**: adaptive bid shading
- RPS minimax: uniform 1/3 mix, expected value = 0 verified numerically
- Colonel Blotto $10 \times 3$: all 66 allocations enumerated, zero-sum verified
- **`kyle_equilibrium(sigma_v, sigma_u)`**: $\lambda = \sigma_v / (2\sigma_u)$, $\beta = \sigma_u/\sigma_v$
- **`kyle_price_discovery(v_true, ...)`** sequential price discovery simulation
- Bid-ask spread derivation from Kyle adverse selection model

# Week 8: Synthesis
- PoA derivations for all 6 settings (Pigou, Braess, HFT, Bertrand, auction, Hawk-Dove)
- HFT PoA growth curve: $O(n)$, unbounded as $n \to \infty$
- Financial market efficiency table: OPEC, equity, IPO, spectrum, HFT, ad auctions

# References

- Nash (1950): Equilibrium Points in n-Person Games
- Vickrey (1961): Counterspeculation, Auctions, and Competitive Sealed Tenders
- Rubinstein (1982): Perfect Equilibrium in a Bargaining Model
- Myerson (1981): Optimal Auction Design
- Roughgarden & Tardos (2002): How Bad is Selfish Routing?
- Kyle (1985): Continuous Auctions and Insider Trading
- Budish, Cramton & Shim (2015): The High-Frequency Trading Arms Race
- Nowak (2006): Evolutionary Dynamics
- Nisan et al. (2007): Algorithmic Game Theory
