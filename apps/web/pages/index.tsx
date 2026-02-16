import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import Sidebar, { DistrictStat } from "../components/Sidebar";

const MapView = dynamic(() => import("../components/MapView"), { ssr: false });

type ColorMode = "rainbow" | "party";

/**
 * base is the URL prefix we fetch:
 *  - "/data" (legacy)
 *  - "/runs/<name>" (static named runs)
 *  - "/outputs/<timestamped_folder>" (latest timestamp runs)
 */
type RunOption = { id: string; label: string; base: string };

const DEFAULT_RUNS: RunOption[] = [
  { id: "latest_data", label: "Legacy Latest (/data)", base: "/data" },

  // These two will be resolved via /outputs/latest.json (we'll override base dynamically)
  { id: "greedy_dem_latest", label: "Greedy (Dem maximize) — latest", base: "/outputs/__LATEST_GREEDY_DEM__" },
  { id: "greedy_rep_latest", label: "Greedy (GOP maximize) — latest", base: "/outputs/__LATEST_GREEDY_REP__" },

  // Optional: still support fixed runs if you keep them
  { id: "kmeans_softcap", label: "KMeans (soft cap) — fixed", base: "/runs/kmeans_softcap" }
];

export default function Home() {
  const [runId, setRunId] = useState<string>("greedy_dem_latest");

  const [latest, setLatest] = useState<Record<string, string> | null>(null);

  const [geojson, setGeojson] = useState<any | null>(null);
  const [districtsGeojson, setDistrictsGeojson] = useState<any | null>(null);
  const [stats, setStats] = useState<DistrictStat[] | null>(null);
  const [hoverDistrict, setHoverDistrict] = useState<number | null>(null);

  const [colorMode, setColorMode] = useState<ColorMode>("rainbow");
  const [showDistrictOutlines, setShowDistrictOutlines] = useState<boolean>(true);
  const [outlineWeight, setOutlineWeight] = useState<number>(2.5);

  const runs = DEFAULT_RUNS;

  // 1) Load /outputs/latest.json once
  useEffect(() => {
    fetch("/outputs/latest.json")
      .then((r) => r.json())
      .then((j) => setLatest(j))
      .catch((e) => {
        console.warn("No /outputs/latest.json yet (run python scripts first). Falling back to /data.", e);
        setLatest({});
      });
  }, []);

  const selectedRun = runs.find((r) => r.id === runId) ?? runs[0];

  // 2) Resolve the base path, swapping in the timestamped folder names
  const base = useMemo(() => {
    const b = selectedRun.base;

    // If we haven't loaded latest.json yet, keep placeholder base (we'll fetch later when latest exists)
    if (!latest) return b;

    // Latest greedy dem
    if (b.includes("__LATEST_GREEDY_DEM__")) {
      const folder = latest["greedy_dem"];
      return folder ? `/outputs/${folder}` : "/data";
    }

    // Latest greedy rep
    if (b.includes("__LATEST_GREEDY_REP__")) {
      const folder = latest["greedy_rep"];
      return folder ? `/outputs/${folder}` : "/data";
    }

    // Fixed base (data or runs)
    return b;
  }, [selectedRun.base, latest]);

  // 3) Fetch map + stats + optional outlines whenever base changes or outlines toggle flips
  useEffect(() => {
    let cancelled = false;

    (async () => {
      setGeojson(null);
      setStats(null);
      setDistrictsGeojson(null);

      try {
        const [gj, st] = await Promise.all([
          fetch(`${base}/map_data.geojson`).then((r) => r.json()),
          fetch(`${base}/district_stats.json`).then((r) => r.json())
        ]);
        if (cancelled) return;
        setGeojson(gj);
        setStats(st);
      } catch (e) {
        console.error("Failed to load run from base:", base, e);
      }

      if (showDistrictOutlines) {
        try {
          const dg = await fetch(`${base}/districts.geojson`).then((r) => r.json());
          if (cancelled) return;
          setDistrictsGeojson(dg);
        } catch (e) {
          // Not fatal if not present
          console.warn("No districts.geojson at base (ok):", base);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [base, showDistrictOutlines]);

  const statsByDistrict = useMemo(() => {
    const m = new Map<number, DistrictStat>();
    (stats ?? []).forEach((s) => m.set(Number((s as any).district), s));
    return m;
  }, [stats]);

  return (
    <div className="appShell">
      <div className="sidebar">
        <div className="headerRow">
          <h2 style={{ margin: 0 }}>Illinois Gerrymandering Demo</h2>
          <span className="badge">Runs</span>
        </div>

        <div className="small" style={{ marginTop: 8 }}>
          Hover precincts to see precinct + district totals. Sidebar highlights hovered district.
        </div>

        {/* Controls */}
        <div style={{ marginTop: 14, padding: 10, border: "1px solid #2a2a2a", borderRadius: 10 }}>
          <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 6 }}>Run</div>
          <select
            value={runId}
            onChange={(e) => setRunId(e.target.value)}
            style={{ width: "100%", padding: 8, borderRadius: 8 }}
          >
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                {r.label}
              </option>
            ))}
          </select>

          <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
            <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 12 }}>
              <input
                type="radio"
                name="colormode"
                checked={colorMode === "rainbow"}
                onChange={() => setColorMode("rainbow")}
              />
              Rainbow districts
            </label>

            <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 12 }}>
              <input
                type="radio"
                name="colormode"
                checked={colorMode === "party"}
                onChange={() => setColorMode("party")}
              />
              Red/Blue winners
            </label>
          </div>

          <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 12, marginTop: 10 }}>
            <input
              type="checkbox"
              checked={showDistrictOutlines}
              onChange={(e) => setShowDistrictOutlines(e.target.checked)}
            />
            Bold district outlines
          </label>

          {showDistrictOutlines ? (
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 12, marginBottom: 4 }}>Outline thickness</div>
              <input
                type="range"
                min={1}
                max={6}
                step={0.5}
                value={outlineWeight}
                onChange={(e) => setOutlineWeight(Number(e.target.value))}
                style={{ width: "100%" }}
              />
            </div>
          ) : null}

          <div className="small" style={{ marginTop: 8 }}>
            Fetching from: <code>{base}</code>
          </div>
        </div>

        <div style={{ marginTop: 12 }}>
          <Sidebar stats={stats ?? []} activeDistrict={hoverDistrict} onHoverDistrict={setHoverDistrict} />
        </div>
      </div>

      <div className="main">
        {!geojson || !stats ? <div className="mapLoading">Loading map data…</div> : null}

        {geojson && stats ? (
          <MapView
            geojson={geojson}
            districtsGeojson={districtsGeojson}
            statsByDistrict={statsByDistrict}
            hoverDistrict={hoverDistrict}
            setHoverDistrict={setHoverDistrict}
            colorMode={colorMode}
            showDistrictOutlines={showDistrictOutlines}
            outlineWeight={outlineWeight}
          />
        ) : null}
      </div>
    </div>
  );
}
