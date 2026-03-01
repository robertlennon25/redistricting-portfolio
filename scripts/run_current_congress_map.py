import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime

from gerry.data.map_pack import load_map_pack
from export_run import export_run

from gerry.algos.district_by_real_map import labels_from_attributes_column

# reuse your helpers if these are in a shared module; otherwise paste them
from run_kmeans_pack import (
    _resolve_pack_dir,
    _resolve_outputs_root,
    _update_latest_manifest,
    _load_pack_arrays,
)


def _resolve_enacted_column(cfg: dict, state: str | None, cli_column: str | None) -> str:
    """
    Priority:
      1) CLI --column
      2) cfg.states.<state>.enacted.congress_col
      3) cfg.algos.current_congress.column (or default CONG_DIST)
    """
    # 3) default from algos.current_congress
    ccfg = ((cfg.get("algos", {}) or {}).get("current_congress", {}) or {})
    column = ccfg.get("column", "CONG_DIST")

    # 2) state override
    if state:
        scfg = (cfg.get("states", {}) or {}).get(state, {}) or {}
        enacted = (scfg.get("enacted", {}) or {})
        column = enacted.get("congress_col", column)

    # 1) CLI override
    if cli_column:
        column = cli_column

    return column


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--state",
        default=None,
        help="Use cfg.states.<state>.assets_dir and write outputs under /outputs/<state>/",
    )
    ap.add_argument(
        "--column",
        default=None,
        help="Override enacted district column in attributes.csv (e.g., 'CONG_DIST' or 'Congress').",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    pack_dir = _resolve_pack_dir(cfg, args.state)
    outputs_root = _resolve_outputs_root(cfg)

    state_key = args.state or (cfg.get("project", {}) or {}).get("default_state") or "default"
    state_outputs_root = outputs_root / state_key
    state_outputs_root.mkdir(parents=True, exist_ok=True)

    pack = load_map_pack(pack_dir)
    unit_ids, coords, weight = _load_pack_arrays(pack_dir, pack)

    column = _resolve_enacted_column(cfg, args.state, args.column)
    print(f"Using enacted column: {column}")

    labels, value_to_label = labels_from_attributes_column(
        pack_dir=pack_dir,
        column=column,
        unit_ids=unit_ids,  # optional safety check if attributes.csv has unit_id
        coerce_int=True,
    )

    # Sanity print
    num_districts = len(set(labels.tolist()))
    print(f"Found {num_districts} districts from '{column}'.")
    print("Example mapping (original -> label):", list(value_to_label.items())[:10])

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"current_congress_{run_id}"
    run_dir = state_outputs_root / folder_name

    export_run(
        pack_dir=pack_dir,
        pack=pack,
        labels=labels,
        run_dir=run_dir,
        title=f"Current Congress Districts ({column}) [{state_key}]",
    )

    _update_latest_manifest(state_outputs_root, "current_congress", folder_name)
    print("Saved:", run_dir)


if __name__ == "__main__":
    main()