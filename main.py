"""
Main entry point: run the checks and experiments, and show the figures.

Usage (from the project folder):
    python main.py           -> real-data analysis (experiment 3)       ~10 s
    python main.py 1         -> experiment 1: static-grid verification  ~30 s
    python main.py 2         -> experiment 2: morphing verification     ~2 min
    python main.py 3         -> experiment 3: real data (first look)
    python main.py 4         -> experiment 4: 5 baseline repeats (repeatability)
    python main.py 5         -> experiment 5: ALL data folders -> tables + plots
    python main.py 6         -> experiment 6: how strength changes during each run
    python main.py 7         -> experiment 7: statistics report (run 5 and 6 first)
    python main.py all       -> tests + experiments 1-7

Each experiment prints its numbers in the terminal, opens a figure window,
and saves the figure in figures/. Close the window to continue.
"""
import os
import runpy
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = {
    "1": "01_static_grid_verification.py",
    "2": "02_morphing_hdmdc.py",
    "3": "03_real_data_first_look.py",
    "4": "04_baseline_repeatability.py",
    "5": "05_all_conditions.py",
    "6": "06_time_course.py",
    "7": "07_statistics.py",
}


def run_experiment(key):
    path = os.path.join(ROOT, "experiments", EXPERIMENTS[key])
    print(f"\n{'=' * 70}\nRunning {EXPERIMENTS[key]}\n{'=' * 70}")
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    runpy.run_path(path, run_name="__main__")


def run_tests():
    print(f"\n{'=' * 70}\nRunning verification tests\n{'=' * 70}")
    subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:warnings"], cwd=ROOT)


if __name__ == "__main__":
    os.chdir(ROOT)
    choice = sys.argv[1] if len(sys.argv) > 1 else "3"
    if choice == "all":
        run_tests()
        for k in EXPERIMENTS:
            run_experiment(k)
    elif choice == "test":
        run_tests()
    elif choice in EXPERIMENTS:
        run_experiment(choice)
    else:
        print(__doc__)
