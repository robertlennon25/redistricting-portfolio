import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import Sidebar, { DistrictStat } from "../components/Sidebar";

const MapView = dynamic(() => import("../components/MapView"), { ssr: false });

type ColorMode = "rainbow" | "party";

/**
 * Run option: base is the URL prefix we fetch.
 * Examples:
 * - "/data" (legacy)
 * - "/outputs/ny/kmeans_softcap_20260227_153000"
 */
type RunOption = { id: string; label: string; base: string };

function prettyRunLabel(key: string) {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function Home() {
  // ----- NEW: state selection -----
  const [states, setStates] = useState<string[] | null>(null);
  const [stateKey, setStateKey] = useState<string | null>(null);

  // ----- existing: run + map state -----
  const [runId, setRunId] = useState("latest_data");
  const [latest, setLatest] = useState<Record<string, string> | null>(null);

  const [geojson, setGeojson] = useState<any>(null);
  const [districtsGeojson, setDistrictsGeojson] = useState<any>(null);
  const [stats, setStats] = useState<DistrictStat[] | null>(null);

  const [hoverDistrict, setHoverDistrict] = useState<number | null>(null);
  const [colorMode, setColorMode] = useState<ColorMode>("rainbow");

  const [showDistrictOutlines, setShowDistrictOutlines] = useState(true);
  const [outlineWeight, setOutlineWeight] = useState(2.5);

  // 0) Load list of states (outputs/states.json)
  useEffect(() => {
    fetch("/outputs/states.json")
      .then((r) => r.json())
      .then((j) => {
        if (Array.isArray(j)) setStates(j.map(String));
        else setStates(["il"]);
      })
      .catch(() => {
        // fallback if file not present yet
        setStates(["il"]);
      });
  }, []);

  // 1) When a state is selected, load its manifest: /outputs/<state>/latest.json
  useEffect(() => {
    if (!stateKey) return;

    setLatest(null);
    setRunId("latest_data");

    fetch(`/outputs/${stateKey}/latest.json`)
      .then((r) => r.json())
      .then((j) => setLatest(j))
      .catch((e) => {
        console.warn(`No /outputs/${stateKey}/latest.json yet.`, e);
        setLatest({});
      });
  }, [stateKey]);

  // 2) Build run options from latest.json (state-scoped)
  const runs: RunOption[] = useMemo(() => {
    const baseRuns: RunOption[] = [
      { id: "latest_data", label: "Legacy Latest (/data)", base: "/data" },
    ];
    if (!latest || !stateKey) return baseRuns;

    const dynamicRuns: RunOption[] = Object.keys(latest)
      .sort()
      .map((key) => {
        const folder = latest[key]; // e.g. "kmeans_softcap_20260227_153000"
        return {
          id: key,
          label: `${prettyRunLabel(key)} — latest`,
          base: folder ? `/outputs/${stateKey}/${folder}` : "/data",
        };
      });

    return baseRuns.concat(dynamicRuns);
  }, [latest, stateKey]);

  // 3) Keep runId valid when manifest loads/changes
  useEffect(() => {
    if (!latest) return;
    const ids = new Set(runs.map((r) => r.id));
    if (!ids.has(runId)) {
      // pick a reasonable default if exists, else legacy
      const preferred =
        (latest["kmeans_softcap"] && "kmeans_softcap") ||
        (latest["k_means_new"] && "k_means_new") ||
        (latest["greedy_dem"] && "greedy_dem") ||
        "latest_data";
      setRunId(preferred);
    }
  }, [latest, runs, runId]);

  const selectedRun = runs.find((r) => r.id === runId) ?? runs[0];
  const base = selectedRun.base;

  // 4) Fetch map + stats + optional outlines whenever base changes (after state chosen)
  useEffect(() => {
    if (!stateKey) return; // don’t fetch anything until state is chosen

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
  }, [base, showDistrictOutlines, stateKey]);

  const statsByDistrict = useMemo(() => {
    const m = new Map<number, DistrictStat>();
    (stats ?? []).forEach((s: any) => m.set(Number(s.district), s));
    return m;
  }, [stats]);

  // ----- NEW: "Back to state select" -----
  function resetToStateSelect() {
    setStateKey(null);
    setLatest(null);
    setRunId("latest_data");
    setGeojson(null);
    setDistrictsGeojson(null);
    setStats(null);
    setHoverDistrict(null);
  }

  // =========================
  // UI: state selection screen
  // =========================
  if (!stateKey) {
    return (
      <div style={{ padding: 24, maxWidth: 920, margin: "0 auto" }}>
        <h2 style={{ marginBottom: 6 }}>Gerrymandering Demo</h2>
        <p style={{ marginTop: 0, opacity: 0.8 }}>
          Choose a state to load its precomputed runs.
        </p>

        {!states ? (
          <div style={{ padding: 12 }}>Loading states…</div>
        ) : (
          <div style={{ display: "grid", gap: 12, gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))" }}>
            {states.map((st) => (
              <button
                key={st}
                onClick={() => setStateKey(st)}
                style={{
                  padding: 16,
                  borderRadius: 12,
                  border: "1px solid rgba(0,0,0,0.12)",
                  background: "white",
                  cursor: "pointer",
                  textAlign: "left",
                }}
              >
                <div style={{ fontWeight: 700, fontSize: 18 }}>{st.toUpperCase()}</div>
                <div style={{ opacity: 0.7, marginTop: 4 }}>View runs</div>
              </button>
            ))}
          </div>
        )}
      </div>
    );
  }

  // =========================
  // UI: main app after state chosen
  // =========================
  return (
    <div style={{ display: "flex", height: "100vh", width: "100%" }}>
      {/* Sidebar */}
      <div style={{ width: 380, borderRight: "1px solid #eee", padding: 16, overflowY: "auto" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <button
            onClick={resetToStateSelect}
            style={{
              padding: "8px 10px",
              borderRadius: 10,
              border: "1px solid rgba(0,0,0,0.12)",
              background: "white",
              cursor: "pointer",
            }}
          >
            ← States
          </button>
          <div>
            <div style={{ fontSize: 18, fontWeight: 800 }}>{stateKey.toUpperCase()}</div>
            <div style={{ opacity: 0.7, fontSize: 12 }}>Pick a run and explore</div>
          </div>
        </div>

        <hr style={{ margin: "14px 0" }} />

        {/* Controls */}
        <div style={{ display: "grid", gap: 12 }}>
          <label style={{ fontWeight: 700 }}>Run</label>
          <select
            value={runId}
            onChange={(e) => setRunId(e.target.value)}
            style={{ width: "100%", padding: 8, borderRadius: 8 }}
            disabled={!latest}
          >
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                {r.label}
              </option>
            ))}
          </select>

          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <input
                type="radio"
                name="colorMode"
                checked={colorMode === "rainbow"}
                onChange={() => setColorMode("rainbow")}
              />
              Rainbow districts
            </label>

            <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <input type="radio" name="colorMode" checked={colorMode === "party"} onChange={() => setColorMode("party")} />
              Red/Blue winners
            </label>
          </div>

          <label style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <input
              type="checkbox"
              checked={showDistrictOutlines}
              onChange={(e) => setShowDistrictOutlines(e.target.checked)}
            />
            Bold district outlines
          </label>

          {showDistrictOutlines ? (
            <div>
              <div style={{ fontWeight: 700, marginBottom: 6 }}>Outline thickness</div>
              <input
                type="range"
                min={0.5}
                max={8}
                step={0.1}
                value={outlineWeight}
                onChange={(e) => setOutlineWeight(Number(e.target.value))}
                style={{ width: "100%" }}
              />
            </div>
          ) : null}

          <div style={{ fontSize: 12, opacity: 0.8 }}>
            Fetching from: <code>{base}</code>
          </div>
        </div>

        <hr style={{ margin: "14px 0" }} />

        <Sidebar
          stats={(stats ?? []) as any}
          activeDistrict={hoverDistrict}
          onHoverDistrict={setHoverDistrict}
        />
      </div>

      {/* Map */}
      <div style={{ flex: 1 }}>
        {!geojson || !stats ? (
          <div style={{ padding: 20 }}>Loading map data…</div>
        ) : (
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
        )}
      </div>
    </div>
  );
}