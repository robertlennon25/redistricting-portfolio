from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

def labels_from_attributes_column(
    pack_dir: Path,
    column: str,
    unit_ids: list[str] | None = None,
    *,
    coerce_int: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Read pack_dir/attributes.csv, use `column` as district assignment,
    and remap unique values -> [0..K-1].

    Returns:
      labels: np.ndarray (N,) ints in [0..K-1]
      value_to_label: dict mapping original district value -> label int
    """
    attrs_path = Path(pack_dir) / "attributes.csv"
    if not attrs_path.exists():
        raise FileNotFoundError(f"Missing {attrs_path}")

    df = pd.read_csv(attrs_path)

    if column not in df.columns:
        raise KeyError(f"attributes.csv missing '{column}'. Available: {list(df.columns)[:50]} ...")

    values = df[column].copy()

    # Clean / normalize
    values = values.replace("", np.nan)
    if values.isna().any():
        # if there are missing districts, fail loudly (better than silently breaking)
        missing = int(values.isna().sum())
        raise ValueError(f"{missing} rows have missing {column}. Fix data or pack build.")

    if coerce_int:
        # common cases: "17", 17, "17.0"
        values = values.astype(str).str.strip()
        values = values.str.replace(r"\.0$", "", regex=True)
        values = values.astype(int)

    # IMPORTANT: ensure alignment with unit_ids if you want an extra safety check
    # (only works if attributes.csv includes the id column used in the pack)
    if unit_ids is not None and "unit_id" in df.columns:
        csv_ids = df["unit_id"].astype(str).tolist()
        if csv_ids != list(unit_ids):
            raise ValueError("attributes.csv order does not match pack unit_ids order.")

    uniq = sorted(pd.unique(values))
    value_to_label = {v: i for i, v in enumerate(uniq)}

    labels = np.array([value_to_label[v] for v in values], dtype=int)
    return labels, value_to_label