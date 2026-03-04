import { useEffect, useMemo, useRef, useState } from "react";

type RunOption = { id: string; label: string; base: string };

function prettyRunLabel(key: string) {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

type ManifestFrame = {
  frame: string;
  step: number;
  seats: number;
  closest_loss: number;
  objective: number;
  locked?: number;
  note?: string;
};

type FlipbookManifest = {
  state: string;
  title: string;
  fps: number;
  frame_every: number;
  frames_dir: string;
  frames: ManifestFrame[];
};

export default function InActionPage() {
  const [states, setStates] = useState<string[] | null>(null);
  const [stateKey, setStateKey] = useState<string | null>(null);

  const [latest, setLatest] = useState<Record<string, string> | null>(null);
  const [runId, setRunId] = useState<string>(""); // ✅ no legacy default

  const [manifest, setManifest] = useState<FlipbookManifest | null>(null);
  const [frameIndex, setFrameIndex] = useState(0);

  const [isPlaying, setIsPlaying] = useState(false);
  const playTimer = useRef<number | null>(null);

  // Load states list
  useEffect(() => {
    fetch("/outputs/states.json")
      .then((r) => r.json())
      .then((j) => {
        if (Array.isArray(j)) setStates(j.map(String));
        else setStates(["il"]);
      })
      .catch(() => setStates(["il"]));
  }, []);

  // When state chosen: load latest.json
  useEffect(() => {
    if (!stateKey) return;

    setLatest(null);
    setRunId("");
    setManifest(null);
    setFrameIndex(0);
    setIsPlaying(false);

    fetch(`/outputs/${stateKey}/latest.json`)
      .then((r) => r.json())
      .then((j) => setLatest(j))
      .catch(() => setLatest({}));
  }, [stateKey]);

  // Build runs list (ONLY hillclimb keys, and no /data fallback)
  const runs: RunOption[] = useMemo(() => {
    if (!latest || !stateKey) return [];

    return Object.keys(latest)
      .filter((key) => key.startsWith("hillclimb"))
      .sort()
      .map((key) => {
        const folder = latest[key];
        return {
          id: key,
          label: `${prettyRunLabel(key)} — latest`,
          base: folder ? `/outputs/${stateKey}/${folder}` : "",
        };
      })
      .filter((r) => !!r.base);
  }, [latest, stateKey]);

  // Ensure runId is valid once runs load
  useEffect(() => {
    if (!runs.length) return;
    if (!runId) setRunId(runs[0].id);
    // if runId no longer exists, reset to first
    const ids = new Set(runs.map((r) => r.id));
    if (runId && !ids.has(runId)) setRunId(runs[0].id);
  }, [runs, runId]);

  const selectedRun = runs.find((r) => r.id === runId) ?? runs[0];
  const base = selectedRun?.base ?? "";

  // Load flipbook manifest for selected run
  useEffect(() => {
    if (!stateKey) return;
    if (!base) return;

    setManifest(null);
    setFrameIndex(0);
    setIsPlaying(false);

    fetch(`${base}/flipbook/manifest.json`)
      .then((r) => {
        if (!r.ok) throw new Error("No flipbook manifest at this run (yet).");
        return r.json();
      })
      .then((j) => setManifest(j))
      .catch((e) => {
        console.warn("Flipbook load failed:", e);
        setManifest(null);
      });
  }, [base, stateKey]);

  // Stop timer when not playing / unmount
  useEffect(() => {
    return () => {
      if (playTimer.current) window.clearInterval(playTimer.current);
      playTimer.current = null;
    };
  }, []);

  // Start/stop playback when isPlaying changes
  useEffect(() => {
    if (!manifest) return;

    if (!isPlaying) {
      if (playTimer.current) window.clearInterval(playTimer.current);
      playTimer.current = null;
      return;
    }

    const fps = manifest.fps || 12;
    const intervalMs = Math.max(20, Math.floor(1000 / fps));

    if (playTimer.current) window.clearInterval(playTimer.current);
    playTimer.current = window.setInterval(() => {
      setFrameIndex((i) => {
        const next = i + 1;
        if (!manifest.frames || next >= manifest.frames.length) return 0;
        return next;
      });
    }, intervalMs);

    return () => {
      if (playTimer.current) window.clearInterval(playTimer.current);
      playTimer.current = null;
    };
  }, [isPlaying, manifest]);

  // UI: state select
  if (!stateKey) {
    return (
      <div style={{ padding: 24, maxWidth: 920, margin: "0 auto" }}>
        <h1 style={{ marginTop: 8 }}>View Redistricting In Action</h1>
        <p style={{ opacity: 0.8 }}>Choose a state, then pick a hillclimb run that has a flipbook recorded.</p>

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
                <div style={{ fontWeight: 800, fontSize: 18 }}>{st.toUpperCase()}</div>
                <div style={{ opacity: 0.7, marginTop: 4 }}>Open flipbook</div>
              </button>
            ))}
          </div>
        )}
      </div>
    );
  }

  const frames = manifest?.frames ?? [];
  const current = frames[frameIndex];

  const imgSrc =
    manifest && current
      ? `${base}/flipbook/${manifest.frames_dir}/${current.frame}`
      : null;

  return (
    <div style={{ padding: 20, maxWidth: 1200, margin: "0 auto" }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
        <div>
          <h1 style={{ marginTop: 8, marginBottom: 6 }}>View Redistricting In Action</h1>
          <div style={{ opacity: 0.8 }}>
            State: <b>{stateKey.toUpperCase()}</b>
          </div>
        </div>

        <button
          onClick={() => {
            setStateKey(null);
            setLatest(null);
            setRunId("");
            setManifest(null);
            setFrameIndex(0);
            setIsPlaying(false);
          }}
          style={{
            height: 40,
            padding: "0 12px",
            borderRadius: 10,
            border: "1px solid rgba(0,0,0,0.12)",
            background: "white",
            cursor: "pointer",
            alignSelf: "center",
          }}
        >
          ← States
        </button>
      </div>

      <div style={{ marginTop: 14, display: "grid", gap: 10 }}>
        <label style={{ fontWeight: 800 }}>Run</label>
        <select
          value={runId}
          onChange={(e) => setRunId(e.target.value)}
          style={{ width: "100%", maxWidth: 680, padding: 10, borderRadius: 10 }}
          disabled={!runs.length}
        >
          {runs.map((r) => (
            <option key={r.id} value={r.id}>
              {r.label}
            </option>
          ))}
        </select>

        <div style={{ fontSize: 12, opacity: 0.75 }}>
          Fetching from: <code>{base || "(select a run)"}</code>
        </div>
      </div>

      <div style={{ marginTop: 18 }}>
        {!manifest ? (
          <div style={{ padding: 14, border: "1px solid rgba(0,0,0,0.12)", borderRadius: 12, background: "white" }}>
            <div style={{ fontWeight: 800, marginBottom: 6 }}>No flipbook found for this run.</div>
            <div style={{ opacity: 0.8, lineHeight: 1.5 }}>
              Make sure the Python run wrote <code>flipbook/manifest.json</code> under the run folder.
            </div>
          </div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 12 }}>
            <div
              style={{
                border: "1px solid rgba(0,0,0,0.12)",
                borderRadius: 14,
                background: "white",
                padding: 16,
              }}
            >
              {/* Image wrapper so it doesn't blow up the layout */}
              {imgSrc ? (
                <div
                  style={{
                    width: "100%",
                    maxWidth: 900,
                    margin: "0 auto",
                    overflow: "auto",
                    borderRadius: 10,
                  }}
                >
                  <img
                    src={imgSrc}
                    alt="Flipbook frame"
                    style={{
                      width: "100%",
                      height: "auto",
                      display: "block",
                      borderRadius: 10,
                    }}
                  />
                </div>
              ) : (
                <div style={{ padding: 24 }}>No frame to display.</div>
              )}

              {/* Controls */}
              <div style={{ marginTop: 12, display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                <button
                  onClick={() => setIsPlaying((p) => !p)}
                  style={{
                    padding: "10px 12px",
                    borderRadius: 10,
                    border: "1px solid rgba(0,0,0,0.12)",
                    background: "white",
                    cursor: "pointer",
                    fontWeight: 800,
                  }}
                >
                  {isPlaying ? "Pause" : "Play"}
                </button>

                <button
                  onClick={() => setFrameIndex((i) => Math.max(0, i - 1))}
                  style={{
                    padding: "10px 12px",
                    borderRadius: 10,
                    border: "1px solid rgba(0,0,0,0.12)",
                    background: "white",
                    cursor: "pointer",
                  }}
                >
                  ← Prev
                </button>

                <button
                  onClick={() => setFrameIndex((i) => Math.min(frames.length - 1, i + 1))}
                  style={{
                    padding: "10px 12px",
                    borderRadius: 10,
                    border: "1px solid rgba(0,0,0,0.12)",
                    background: "white",
                    cursor: "pointer",
                  }}
                >
                  Next →
                </button>

                <div style={{ opacity: 0.8, fontSize: 13 }}>
                  Frame <b>{frameIndex + 1}</b> / <b>{frames.length}</b>
                  {current ? (
                    <>
                      {" "}
                      — step <b>{current.step}</b>, seats <b>{current.seats}</b>, closest_loss{" "}
                      <b>{Math.round(current.closest_loss)}</b>
                      {typeof current.locked === "number" ? (
                        <>
                          {" "}
                          , locked <b>{current.locked}</b>
                        </>
                      ) : null}
                    </>
                  ) : null}
                </div>
              </div>

              {/* Slider */}
              <div style={{ marginTop: 10 }}>
                <input
                  type="range"
                  min={0}
                  max={Math.max(0, frames.length - 1)}
                  step={1}
                  value={frameIndex}
                  onChange={(e) => setFrameIndex(Number(e.target.value))}
                  style={{ width: "100%" }}
                />
              </div>

              {/* Text below slider */}
              <div style={{ marginTop: 10, fontSize: 14, lineHeight: 1.55, opacity: 0.85 }}>
                <b>What you’re seeing:</b> each frame shows the district assignment as hillclimb runs. The algorithm
                proposes boundary moves that respect contiguity and population constraints, aiming to increase the
                number of seats won while improving near-flip margins. If a district flips to a win, the frame can
                highlight it (depending on your recorder settings).
              </div>

              {current?.note ? (
                <div style={{ marginTop: 10, fontSize: 13, opacity: 0.8 }}>
                  <b>Note:</b> {current.note}
                </div>
              ) : null}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}