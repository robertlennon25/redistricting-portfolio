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

function isDemWinner(w: string | undefined) {
  const ww = String(w ?? "").toLowerCase();
  return ww.includes("dem");
}

function isRepWinner(w: string | undefined) {
  const ww = String(w ?? "").toLowerCase();
  return ww.includes("rep") || ww.includes("gop");
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

  const summary = useMemo(() => {
    let demDistricts = 0;
    let repDistricts = 0;
    let demVotes = 0;
    let repVotes = 0;

    for (const s of stats) {
      if (isDemWinner(s.winner)) demDistricts += 1;
      else if (isRepWinner(s.winner)) repDistricts += 1;

      demVotes += Number(s.dem_votes ?? 0);
      repVotes += Number(s.rep_votes ?? 0);
    }

    return { demDistricts, repDistricts, demVotes, repVotes };
  }, [stats]);

  return (
    <div style={{ display: "grid", gap: 12 }}>
      {/* Summary block (above Run on the map page because Sidebar is rendered below Run controls) */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 12,
          padding: 12,
          borderRadius: 12,
          border: "1px solid rgba(0,0,0,0.10)",
          background: "rgba(0,0,0,0.02)"
        }}
      >
        <div style={{ textAlign: "center" }}>
          <div style={{ fontWeight: 900, fontSize: 13, color: "#2b6fff" }}>
            Democratic Districts
          </div>
          <div style={{ fontWeight: 900, fontSize: 26, marginTop: 2 }}>
            {summary.demDistricts}
          </div>
          <div style={{ opacity: 0.75, fontSize: 12, marginTop: 2 }}>
            Total votes: {fmt(summary.demVotes)}
          </div>
        </div>

        <div style={{ textAlign: "center" }}>
          <div style={{ fontWeight: 900, fontSize: 13, color: "#ff3b3b" }}>
            Republican Districts
          </div>
          <div style={{ fontWeight: 900, fontSize: 26, marginTop: 2 }}>
            {summary.repDistricts}
          </div>
          <div style={{ opacity: 0.75, fontSize: 12, marginTop: 2 }}>
            Total votes: {fmt(summary.repVotes)}
          </div>
        </div>
      </div>

      {/* Existing table */}
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
                <td>
                  {fmt(Number(s.margin))} ({fmtPct(Number(s.margin_pct))})
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}