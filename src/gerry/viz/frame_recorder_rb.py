from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt


@dataclass
class FrameMeta:
    step: int
    seats: int
    closest_loss: float
    objective: float
    locked: int = 0
    note: str = ""


class FrameRecorder:
    """
    Writes a "flipbook" folder:
      <run_dir>/flipbook/              (default)
        frames/
          frame_000000.png
          ...
        manifest.json

    NEW:
      - color_mode="party" writes to <run_dir>/flipbook_party/ (does NOT overwrite flipbook/)
      - party colors: precinct fill is red/blue based on CURRENT district winner each frame
      - change highlighting: bold outline around the district that changed this step
    """

    def __init__(
        self,
        *,
        pack_dir: Path,
        run_dir: Path,
        state: str,
        title: str = "",
        dpi: int = 140,
        figsize: tuple[float, float] = (9.5, 9.5),
        facecolor: str = "white",
        bounds_pad_frac: float = 0.01,  # 1% padding around bounds
        highlight_linewidth: float = 2.8,
        color_mode: str = "party",  # "rainbow" (old) or "party" (new)
    ):
        self.pack_dir = Path(pack_dir)
        self.run_dir = Path(run_dir)
        self.state = state
        self.title = title or f"Redistricting ({state.upper()})"
        self.dpi = dpi
        self.figsize = figsize
        self.facecolor = facecolor
        self.bounds_pad_frac = float(bounds_pad_frac)
        self.highlight_linewidth = float(highlight_linewidth)
        self.color_mode = str(color_mode)

        # --- Output dirs ---
        # Never overwrite existing flipbook. If color_mode="party", write to flipbook_party.
        self.flipbook_dir = self.run_dir / ("flipbook_party" if self.color_mode == "party" else "flipbook")
        self.frames_dir = self.flipbook_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        # --- Load geometry + id mapping ---
        shapes_path = self.pack_dir / "shapes.geojson"
        if not shapes_path.exists():
            raise FileNotFoundError(f"Missing shapes.geojson at {shapes_path}")

        id_to_idx_path = self.pack_dir / "id_to_idx.json"
        if not id_to_idx_path.exists():
            raise FileNotFoundError(f"Missing id_to_idx.json at {id_to_idx_path}")

        id_to_idx = json.loads(id_to_idx_path.read_text())
        self.id_to_idx = {str(k): int(v) for k, v in id_to_idx.items()}

        gdf = gpd.read_file(shapes_path)
        if "unit_id" not in gdf.columns:
            raise KeyError(f"{shapes_path} missing 'unit_id' column.")

        gdf["unit_id"] = gdf["unit_id"].astype(str)
        gdf["idx"] = gdf["unit_id"].map(self.id_to_idx)

        if gdf["idx"].isna().any():
            missing = int(gdf["idx"].isna().sum())
            raise ValueError(f"{missing} shapes have unit_id not found in id_to_idx.json")

        gdf["idx"] = gdf["idx"].astype(int)
        self.gdf = gdf

        # Fixed camera bounds (prevents varying crop/scale)
        minx, miny, maxx, maxy = self.gdf.total_bounds
        dx = (maxx - minx) * self.bounds_pad_frac
        dy = (maxy - miny) * self.bounds_pad_frac
        self._bounds = (minx - dx, miny - dy, maxx + dx, maxy + dy)

        self.frames: list[Dict[str, Any]] = []

        # Flip highlighting state (existing behavior: newly-won districts)
        self._prev_wins: Optional[np.ndarray] = None  # (K,) bool

        # NEW: store previous labels to detect which district changed this step
        self._prev_labels: Optional[np.ndarray] = None

        # NEW: load dem/rep votes for party coloring
        attrs_path = self.pack_dir / "attributes.csv"
        if not attrs_path.exists():
            raise FileNotFoundError(f"Missing attributes.csv at {attrs_path}")
        import pandas as pd
        attrs = pd.read_csv(attrs_path)
        if "dem_votes" not in attrs.columns or "rep_votes" not in attrs.columns:
            raise KeyError("attributes.csv must contain dem_votes and rep_votes columns")
        self._dem_votes = attrs["dem_votes"].astype(float).values
        self._rep_votes = attrs["rep_votes"].astype(float).values

    def _frame_path(self, frame_no: int) -> Path:
        return self.frames_dir / f"frame_{frame_no:06d}.png"

    @staticmethod
    def _party_colors_for_districts(dem_sum: np.ndarray, rep_sum: np.ndarray) -> np.ndarray:
        """
        Returns per-district color strings: Dem=blue, Rep=red, tie=gray.
        """
        out = np.empty(len(dem_sum), dtype=object)
        for i in range(len(dem_sum)):
            d = float(dem_sum[i])
            r = float(rep_sum[i])
            if d > r:
                out[i] = "#2b6fff"
            elif r > d:
                out[i] = "#ff3b3b"
            else:
                out[i] = "#999999"
        return out

    def _compute_party_facecolors(self, labels_full: np.ndarray) -> np.ndarray:
        """
        Compute per-unit facecolors (len N) based on current district winner.
        """
        labels_full = np.asarray(labels_full).astype(int)
        K = int(labels_full.max() + 1) if labels_full.size else 0

        dem_sum = np.zeros(K, dtype=float)
        rep_sum = np.zeros(K, dtype=float)
        for d in range(K):
            mask = labels_full == d
            if np.any(mask):
                dem_sum[d] = float(self._dem_votes[mask].sum())
                rep_sum[d] = float(self._rep_votes[mask].sum())

        district_colors = self._party_colors_for_districts(dem_sum, rep_sum)
        return district_colors[labels_full]

    def _changed_district(self, labels_full: np.ndarray) -> Optional[int]:
        """
        Determine which district changed this step (most common destination label among changed units).
        """
        if self._prev_labels is None:
            return None
        prev = self._prev_labels
        if prev.shape != labels_full.shape:
            return None

        diff = np.where(prev != labels_full)[0]
        if diff.size == 0:
            return None

        moved_to = labels_full[diff]
        vals, counts = np.unique(moved_to, return_counts=True)
        return int(vals[np.argmax(counts)])

    def record(
        self,
        *,
        frame_no: int,
        labels: np.ndarray,
        meta: FrameMeta,
        margins: Optional[Sequence[float]] = None,  # party margins per district (len=K)
        edgecolor: Optional[str] = None,
        linewidth: float = 0.10,
        # NEW: allow explicit override, but default is "district changed this step"
        highlight_district: Optional[int] = None,
        highlight_edgecolor: str = "black",
        highlight_linewidth: Optional[float] = None,
    ) -> Path:
        """
        Render and write one frame.

        margins:
          If provided, used to detect districts that just flipped from losing->winning
          (margin <= 0 -> margin > 0). Those districts get a bold black outline AND a note.

        highlight_district:
          If provided, bold-outline that district.
          If None, we auto-detect the district that changed from previous labels and outline it.
        """
        # Attach district labels by idx (fast)
        idx = self.gdf["idx"].to_numpy()
        labels_full = np.asarray(labels).astype(int)
        self.gdf["district"] = labels_full[idx].astype(int)

        # Determine which district changed (auto) unless caller overrides
        auto_changed = self._changed_district(labels_full)
        if highlight_district is None:
            highlight_district = auto_changed

        # Determine newly-won districts to highlight (existing feature)
        highlight_flip_districts: list[int] = []
        if margins is not None:
            m = np.asarray(list(margins), dtype=float)
            wins = m > 0
            if self._prev_wins is None:
                self._prev_wins = wins.copy()
            else:
                new_wins = np.where(wins & ~self._prev_wins)[0]
                highlight_flip_districts = [int(x) for x in new_wins.tolist()]
                self._prev_wins = wins.copy()

            if highlight_flip_districts and not meta.note:
                meta.note = "Flip(s): " + ", ".join(map(str, highlight_flip_districts))

        # --- Color ---
        fig, ax = plt.subplots(figsize=self.figsize)
        fig.patch.set_facecolor(self.facecolor)

        if self.color_mode == "party":
            facecolors_full = self._compute_party_facecolors(labels_full)
            self.gdf["_face"] = facecolors_full[idx]

            # Base plot with explicit colors (no categorical colormap)
            self.gdf.plot(
                ax=ax,
                color=self.gdf["_face"],
                legend=False,
                linewidth=linewidth,
                edgecolor=edgecolor,
            )
        else:
            # Old behavior: categorical district coloring
            self.gdf.plot(
                ax=ax,
                column="district",
                categorical=True,
                legend=False,
                linewidth=linewidth,
                edgecolor=edgecolor,
            )

        # --- Highlight: district changed this step ---
        if highlight_district is not None:
            hi = self.gdf[self.gdf["district"] == int(highlight_district)]
            if len(hi) > 0:
                hi_diss = hi.dissolve(by="district")
                hi_diss.boundary.plot(
                    ax=ax,
                    linewidth=float(highlight_linewidth if highlight_linewidth is not None else self.highlight_linewidth),
                    color=highlight_edgecolor,
                    alpha=0.95,
                )

        # --- Highlight: newly-won flips (existing) ---
        if highlight_flip_districts:
            hi2 = self.gdf[self.gdf["district"].isin(highlight_flip_districts)]
            if len(hi2) > 0:
                hi2_diss = hi2.dissolve(by="district")
                hi2_diss.boundary.plot(ax=ax, linewidth=self.highlight_linewidth, color="black", alpha=0.95)

        # Fixed camera
        minx, miny, maxx, maxy = self._bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()

        # Text overlay
        overlay = (
            f"{self.title}\n"
            f"step={meta.step} | seats={meta.seats} | closest_loss={meta.closest_loss:.1f} | "
            f"obj={meta.objective:.2f} | locked={meta.locked}"
        )
        if meta.note:
            overlay += f"\n{meta.note}"
        if highlight_district is not None:
            overlay += f"\nChanged district: {int(highlight_district)}"

        ax.text(
            0.01,
            0.01,
            overlay,
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.80),
        )

        out_path = self._frame_path(frame_no)

        # IMPORTANT: no bbox_inches="tight" (prevents per-frame auto-cropping/scale changes)
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)

        self.frames.append(
            {
                "frame": out_path.name,
                "step": int(meta.step),
                "seats": int(meta.seats),
                "closest_loss": float(meta.closest_loss),
                "objective": float(meta.objective),
                "locked": int(meta.locked),
                "note": meta.note,
                "changed_district": int(highlight_district) if highlight_district is not None else None,
            }
        )

        # Update prev labels for next step
        self._prev_labels = labels_full.copy()

        return out_path

    def write_manifest(self, *, fps: int = 12, frame_every: int = 5) -> Path:
        manifest = {
            "state": self.state,
            "title": self.title,
            "fps": int(fps),
            "frame_every": int(frame_every),
            "frames_dir": "frames",
            "frames": self.frames,
            "color_mode": self.color_mode,
        }
        out_path = self.flipbook_dir / "manifest.json"

        # Never overwrite: if exists, suffix with _v2/_v3...
        if out_path.exists():
            base = self.flipbook_dir / "manifest"
            k = 2
            while True:
                cand = Path(str(base) + f"_v{k}.json")
                if not cand.exists():
                    out_path = cand
                    break
                k += 1

        out_path.write_text(json.dumps(manifest, indent=2))
        return out_path