from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import run_experiment
from .visualize import render_policy_frontier


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LLM inference scheduling policies.")
    parser.add_argument("--requests", type=int, default=160)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=Path("reports/baseline.json"))
    parser.add_argument("--visual", type=Path, default=Path("assets/policy-frontier.svg"))
    args = parser.parse_args()

    report = run_experiment(args.requests, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.visual.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.visual.write_text(render_policy_frontier(report) + "\n", encoding="utf-8")
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    main()
