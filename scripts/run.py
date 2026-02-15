from __future__ import annotations
import argparse
from pathlib import Path
import yaml

def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    algo = cfg["run"]["algo"]
    if algo == "greedy_packing":
        from gerry.algos.greedy_packing import run_greedy_packing
        run_greedy_packing(cfg)
    else:
        raise ValueError(f"Unknown algo: {algo}")

if __name__ == "__main__":
    main()
