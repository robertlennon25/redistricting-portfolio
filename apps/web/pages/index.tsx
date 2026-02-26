import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import Sidebar, { DistrictStat } from "../components/Sidebar";

const MapView = dynamic(() => import("../components/MapView"), { ssr: false });

type ColorMode = "rainbow" | "party";

/**
 * base is the URL prefix we fetch:
 *  - "/data" (legacy)
 *  - "/outputs/<timestamped_folder>" (latest timestamp runs)
 */
type RunOption = { id: string; label: string; base: string };

// Pretty label from key like "greedy2_dem" -> "Greedy2 Dem"
function prettyRunLabel(key: string) {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function Home() {
  const [runId, setRunId] = useState<string>("latest_data");
  const [latest, setLatest] = useState<Record<string, string> | null>(null);

  const [geojson, setGeojson] = useState<any | null>(null);
  const [districtsGeojson, setDistrictsGeojson] = useState<any | null>(null);
  const [stats, setStats] = useState<DistrictStat[] | null>(null);
  const [hoverDistrict, setHoverDistrict] = useState<number | null>(null);

  const [colorMode, setColorMode] = useState<ColorMode>("rainbow");
  const [showDistrictOutlines, setShowDistrictOutlines] = useState<boolean>(true);
  const [outlineWeight, setOutlineWeight] = useState<number>(2.5);

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

  // 2) Build run options dynamically from latest.json
  const runs: RunOption[] = useMemo(() => {
    const baseRuns: RunOption[] = [{ id: "latest_data", label: "Legacy Latest (/data)", base: "/data" }];

    if (!latest) return baseRuns;

    const dynamicRuns: RunOption[] = Object.keys(latest)
      .sort()
      .map((key) => {
        const folder = latest[key];
        return {
          id: key, // runId equals key in latest.json
          label: `${prettyRunLabel(key)} — latest`,
          base: folder ? `/outputs/${folder}` : "/data",
        };
      });

    return baseRuns.concat(dynamicRuns);
  }, [latest]);

  // 3) Keep runId valid when latest.json loads/changes
  useEffect(() => {
    if (!latest) return;

    const ids = new Set(runs.map((r) => r.id));
    if (!ids.has(runId)) {
      const preferred =
        (latest["greedy2_dem"] && "greedy2_dem") ||
        (latest["greedy_dem"] && "greedy_dem") ||
        "latest_data";
      setRunId(preferred);
    }
  }, [latest, runs, runId]);

  const selectedRun = runs.find((r) => r.id === runId) ?? runs[0];
  const base = selectedRun.base;

  // 4) Fetch map + stats + optional outlines whenever base changes or outlines toggle flips
  useEffect(() => {
    let cancelled = false;

    (async () => {
      setGeojson(null);
      setStats(null);
      setDistrictsGeojson(null);

      try {
        const [gj, st] = await Promise.all([
          fetch(`${base}/map_data.geojson`).then((r) => r.json()),
          fetch(`${base}/district_stats.json`).then((r) => r.json()),
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