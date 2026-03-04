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
      <run_dir>/flipbook/
        frames/
          frame_000000.png
          frame_000001.png
          ...
        manifest.json

    Enhancements:
      - fixed camera (prevents "skinny frames"): fixed x/y limits + no bbox_inches="tight"
      - optional flip highlighting: bold boundary for districts that just flipped to a win
        (requires margins list passed to record())
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

        self.flipbook_dir = self.run_dir / "flipbook"
        self.frames_dir = self.flipbook_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

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

        # Flip highlighting state
        self._prev_wins: Optional[np.ndarray] = None  # (K,) bool

    def _frame_path(self, frame_no: int) -> Path:
        return self.frames_dir / f"frame_{frame_no:06d}.png"

    def record(
        self,
        *,
        frame_no: int,
        labels: np.ndarray,
        meta: FrameMeta,
        margins: Optional[Sequence[float]] = None,  # party margins per district (len=K)
        edgecolor: Optional[str] = None,
        linewidth: float = 0.10,
    ) -> Path:
        """
        Render and write one frame.

        margins:
          If provided, used to detect districts that just flipped from losing->winning
          (margin <= 0 -> margin > 0). Those districts get a bold black outline.
        """
        # Attach district labels by idx (fast)
        idx = self.gdf["idx"].to_numpy()
        self.gdf["district"] = labels[idx].astype(int)

        # Determine newly-won districts to highlight
        highlight_districts: list[int] = []
        if margins is not None:
            m = np.asarray(list(margins), dtype=float)
            wins = m > 0
            if self._prev_wins is None:
                self._prev_wins = wins.copy()
            else:
                new_wins = np.where(wins & ~self._prev_wins)[0]
                highlight_districts = [int(x) for x in new_wins.tolist()]
                self._prev_wins = wins.copy()

            if highlight_districts and not meta.note:
                meta.note = "Flip(s): " + ", ".join(map(str, highlight_districts))

        fig, ax = plt.subplots(figsize=self.figsize)
        fig.patch.set_facecolor(self.facecolor)

        # Base plot
        self.gdf.plot(
            ax=ax,
            column="district",
            categorical=True,
            legend=False,
            linewidth=linewidth,
            edgecolor=edgecolor,  # None = fastest; or "black"
        )

        # Highlight outlines (dissolve just the few districts)
        if highlight_districts:
            hi = self.gdf[self.gdf["district"].isin(highlight_districts)]
            if len(hi) > 0:
                hi_diss = hi.dissolve(by="district")
                hi_diss.boundary.plot(ax=ax, linewidth=self.highlight_linewidth, color="black")

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
            }
        )

        return out_path

    def write_manifest(self, *, fps: int = 12, frame_every: int = 5) -> Path:
        manifest = {
            "state": self.state,
            "title": self.title,
            "fps": int(fps),
            "frame_every": int(frame_every),
            "frames_dir": "frames",
            "frames": self.frames,
        }
        out_path = self.flipbook_dir / "manifest.json"
        out_path.write_text(json.dumps(manifest, indent=2))
        return out_path