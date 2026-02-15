import React, { useMemo } from "react";

export type DistrictStat = {
  district: number;
  dem_votes: number;
  rep_votes: number;
  weight: number;
  winner: string;
  margin: number;
  margin_pct: number;
};

function fmt(n: number) {
  if (!Number.isFinite(n)) return "";
  return Math.round(n).toLocaleString();
}

function fmtPct(n: number) {
  if (!Number.isFinite(n)) return "";
  return `${n.toFixed(1)}%`;
}

function colorForDistrict(d: number) {
  // deterministic bright-ish hue
  const hue = (d * 47) % 360;
  return `hsl(${hue} 70% 55%)`;
}

export default function Sidebar({
  stats,
  activeDistrict,
  onHoverDistrict
}: {
  stats: DistrictStat[];
  activeDistrict: number | null;
  onHoverDistrict: (d: number | null) => void;
}) {
  const sorted = useMemo(() => [...stats].sort((a, b) => a.district - b.district), [stats]);

  return (
    <table className="table">
      <thead>
        <tr>
          <th>District</th>
          <th>Winner</th>
          <th>Dem</th>
          <th>GOP</th>
          <th>Pop</th>
          <th>Margin</th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((s) => {
          const d = Number(s.district);
          const active = activeDistrict === d;
          return (
            <tr
              key={d}
              className={active ? "rowActive" : undefined}
              onMouseEnter={() => onHoverDistrict(d)}
              onMouseLeave={() => onHoverDistrict(null)}
              style={{ cursor: "default" }}
            >
              <td>
                <span className="colorDot" style={{ background: colorForDistrict(d) }} />
                {d}
              </td>
              <td>{s.winner}</td>
              <td>{fmt(Number(s.dem_votes))}</td>
              <td>{fmt(Number(s.rep_votes))}</td>
              <td>{fmt(Number(s.weight))}</td>
              <td>{fmt(Number(s.margin))} ({fmtPct(Number(s.margin_pct))})</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
