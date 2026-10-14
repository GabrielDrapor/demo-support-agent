"""
Send demo traffic to the stable version and to fault versions at the same time.

Each fault is a real commit on its own `fault/*` branch. This script checks
out each chosen fault branch into a git worktree under `.worktrees/` and runs
the traffic driver there, so its traces carry that commit as the git ref.
The rest of the sessions run on the current checkout (the stable version).

Usage:
    python run_demo.py --sessions 40 --faults f1,f3 --fault-share 0.15
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKTREES = ROOT / ".worktrees"

FAULTS = {
    "f1": "fault/f1-us-date-format",
    "f2": "fault/f2-draft-filter",
    "f3": "fault/f3-refund-limit-config",
    "f4": "fault/f4-no-customer-cache",
    "f5": "fault/f5-bigger-agent-model",
    "f6": "fault/f6-chained-handoffs",
}


def _git(*args: str, cwd: Path = ROOT) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def checkout_fault(branch: str) -> Path:
    """Put the latest commit of a fault branch into its own worktree."""
    path = WORKTREES / branch.replace("/", "-")
    _git("fetch", "--quiet", "origin", branch)
    if path.exists():
        _git("checkout", "--quiet", "--detach", f"origin/{branch}", cwd=path)
    else:
        _git("worktree", "add", "--quiet", "--detach", str(path), f"origin/{branch}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=40, help="total sessions for this run")
    parser.add_argument("--faults", default="", help="comma-separated fault IDs, e.g. f1,f3")
    parser.add_argument("--fault-share", type=float, default=0.15)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    faults = [f for f in args.faults.split(",") if f]
    unknown = [f for f in faults if f not in FAULTS]
    if unknown:
        parser.error(f"unknown faults: {', '.join(unknown)}")

    per_fault = round(args.sessions * args.fault_share / len(faults)) if faults else 0
    runs = [(ROOT, args.sessions - per_fault * len(faults))]
    runs += [(checkout_fault(FAULTS[f]), per_fault) for f in faults if per_fault > 0]

    procs = []
    for cwd, sessions in runs:
        if sessions <= 0:
            continue
        print(f"{sessions:>4} sessions on {cwd.name}")
        cmd = [sys.executable, "traffic_driver.py", "--sessions", str(sessions)]
        cmd += ["--concurrency", str(args.concurrency)]
        procs.append(subprocess.Popen(cmd, cwd=cwd))
    sys.exit(max(p.wait() for p in procs))


if __name__ == "__main__":
    main()
