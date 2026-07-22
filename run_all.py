import os
import sys
import time
import subprocess

SCRIPTS = [
    ("Week 1", "week1_foundations.py",            "PD, payoff matrices, dominance, OPEC Cournot preview"),
    ("Week 2", "week2_nash_competition.py",        "Nash solver, BRD, Bertrand, Cournot"),
    ("Week 3", "week3_auctions.py",                "1st/2nd price auctions, Revenue Equivalence, real world"),
    ("Week 4", "week4_sequential_bargaining.py",   "Backward induction, Rubinstein bargaining, M&A"),
    ("Week 5", "week5_congestion_poa.py",          "Pigou PoA=4/3, Braess paradox, HFT arms race"),
    ("Week 6", "week6_evolutionary.py",            "Replicator dynamics, Hawk Dove ESS, trading strategies"),
    ("Week 7", "week7_multiagent_kyle.py",         "Repeated auctions, minimax, Blotto, Kyle model"),
    ("Week 8", "week8_synthesis.py",               "Cross game PoA synthesis, financial market efficiency"),
]

os.makedirs("figures", exist_ok=True)

print("=" * 65)
print(" SoS Game Theory: Full Pipeline ")
print("=" * 65)

total_start = time.time()
for label, script, description in SCRIPTS:
    print(f"\n{''*65}")
    print(f"  {label}: {script}")
    print(f"  {description}")
    print(f"{''*65}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed (return code {result.returncode})")
        sys.exit(1)
    print(f"  Done in {time.time() - t0:.1f}s")

total_time = time.time() - total_start
print(f"\n{'='* 65}")
print(f"  All 8 weeks complete in {total_time:.1f}s")
print(f"  Figures saved to ./figures/")
print(f"{'='* 65}")
