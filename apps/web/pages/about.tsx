// app/about/page.tsx (or wherever your AboutPage lives)
export default function AboutPage() {
  return (
    <div style={{ padding: 24, maxWidth: 1080, margin: "0 auto" }}>
      {/* ---------- HERO ---------- */}
      <header style={{ marginTop: 8, marginBottom: 18 }}>
        <h1 style={{ margin: 0, fontSize: 40, letterSpacing: -0.5 }}>About</h1>
        <p style={{ marginTop: 10, opacity: 0.82, lineHeight: 1.7, fontSize: 16, maxWidth: 900 }}>
          This site visualizes congressional redistricting runs generated offline (K-Means, contiguity enforcement,
          hill-climbing optimization, and more). Each run exports a <b>GeoJSON map</b> plus <b>district statistics</b>,
          which the web app loads as static files — no server required.
        </p>
      </header>

      {/* ---------- QUICK FACTS ---------- */}
      <section
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
          gap: 12,
          marginBottom: 22,
        }}
      >
        {[
          {
            title: "What you can do",
            body: "Browse runs by state, inspect districts, and compare algorithm outputs side-by-side.",
          },
          {
            title: "What powers it",
            body: "Offline scripts export map + stats into /public/outputs so the frontend can fetch them directly.",
          },
          {
            title: "Core constraints",
            body: "Population balance + contiguity are enforced while optimizing for partisan seats.",
          },
        ].map((c) => (
          <div
            key={c.title}
            style={{
              border: "1px solid rgba(255,255,255,0.12)",
              borderRadius: 16,
              padding: 14,
              background: "rgba(255,255,255,0.03)",
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: 6 }}>{c.title}</div>
            <div style={{ opacity: 0.82, lineHeight: 1.6 }}>{c.body}</div>
          </div>
        ))}
      </section>

      {/* ---------- SECTION: WHAT YOU'RE LOOKING AT ---------- */}
      <section style={{ marginBottom: 26 }}>
        <h2 style={{ margin: "18px 0 10px", fontSize: 22 }}>What you’re looking at</h2>

        <div style={{ display: "grid", gridTemplateColumns: "1.2fr 0.8fr", gap: 16 }}>
          <div
            style={{
              border: "1px solid rgba(255,255,255,0.12)",
              borderRadius: 16,
              padding: 16,
              background: "rgba(255,255,255,0.03)",
            }}
          >
            <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.7, opacity: 0.88 }}>
              <li>
                <b>Runs</b>: precomputed districting outputs (different algorithms / settings) for each state.
              </li>
              <li>
                <b>Map</b>: the GeoJSON for the selected run, rendered in the browser.
              </li>
              <li>
                <b>Flipbook</b>: step-by-step snapshots of an algorithm evolving a map over time.
              </li>
              <li>
                <b>Stats</b>: per-district and statewide summaries exported alongside the GeoJSON.
              </li>
            </ul>
          </div>

          {/* Optional: tiny legend / image slot */}
          <div
            style={{
              border: "1px dashed rgba(255,255,255,0.22)",
              borderRadius: 16,
              padding: 16,
              background: "rgba(255,255,255,0.02)",
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: 8 }}>How To Switch Runs</div>
            <div style={{ opacity: 0.75, lineHeight: 1.6 }}>
              Go to "Map", select a state, then view runs.
            </div>
                  <div
        style={{
          border: "1px solid rgba(0,0,0,0.12)",
          borderRadius: 16,
          overflow: "hidden",
          background: "rgba(0,0,0,0.02)",
        }}
      >
        <div style={{ padding: 12, display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
          <div style={{ fontWeight: 800 }}>Demo: Switching runs</div>
          <div style={{ fontSize: 12, opacity: 0.7 }}>Mouseover + click between runs</div>
        </div>

  <video
    controls
    muted
    playsInline
    preload="metadata"
    style={{ width: "100%", height: "auto", display: "block" }}
    poster="/about/demo_poster.png" // optional
  >
    <source src="/about/0304.mp4" type="video/webm" />
    {/* <source src="0304.mp4" type="video/mp4" /> */}
    Your browser does not support the video tag.
  </video>
</div>
          </div>
        </div>
      </section>

      {/* ---------- SECTION: PIPELINE ---------- */}
      <section style={{ marginBottom: 26 }}>
        <h2 style={{ margin: "18px 0 10px", fontSize: 22 }}>Pipeline</h2>

        <div
          style={{
            border: "1px solid rgba(255,255,255,0.12)",
            borderRadius: 16,
            padding: 16,
            background: "rgba(255,255,255,0.03)",
          }}
        >
          <ol style={{ margin: 0, paddingLeft: 18, lineHeight: 1.75, opacity: 0.88 }}>
            <li>
              <b>Load state data</b> (precinct geometry + population + election results).
            </li>
            <li>
              <b>K-Means initialization</b> to form a fast, compact baseline map.
            </li>
            <li>
              <b>Contiguity enforcement</b> to eliminate disconnected “islands.”
            </li>
            <li>
              <b>Hill-climb optimization</b> to improve seat outcomes while respecting constraints.
            </li>
            <li>
              <b>Export artifacts</b> (GeoJSON + stats + optional flipbook frames) into <code>/public/outputs</code>.
            </li>
          </ol>
        </div>

        <div style={{ marginTop: 10, opacity: 0.75, lineHeight: 1.6 }}>
          The frontend is intentionally simple: it fetches the exported files and renders them. That makes deployment
          easy (static hosting) and keeps the heavy compute offline.
        </div>
      </section>

      {/* ---------- SECTION: KMEANS vs CURRENT DISTRICTS ---------- */}
      <section style={{ marginBottom: 26 }}>
        <h2 style={{ margin: "18px 0 10px", fontSize: 22 }}>K-Means vs. Current Districts</h2>
        <p style={{ opacity: 0.82, lineHeight: 1.7, marginTop: 0 }}>
          A useful baseline comparison is <b>current enacted districts</b> vs. a purely geographic clustering baseline
          from <b>K-Means</b>. K-Means tends to produce compact shapes quickly, but can violate contiguity before
          postprocessing.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
          {/* Current */}
          <div
            style={{
              border: "1px solid rgba(255,255,255,0.12)",
              borderRadius: 16,
              padding: 14,
              background: "rgba(255,255,255,0.03)",
            }}
          >
            <div style={{ fontWeight: 800, marginBottom: 8 }}>Current (Enacted) Map</div>
            <div style={{ opacity: 0.78, lineHeight: 1.6, marginBottom: 10 }}>
              The official district plan used for elections.
            </div>
            <img
              src="/about/current_map.png"
              alt="Current districts"
              style={{
                width: "100%",
                height: "auto",
                objectFit: "cover",
                borderRadius: 12,
                border: "1px solid rgba(255,255,255,0.12)",
              }}
              onError={(e) => {
                (e.currentTarget as HTMLImageElement).style.display = "none";
              }}
            />
            <div style={{ marginTop: 10, opacity: 0.7, fontSize: 13 }}>
              <code>Current Congressional Map in New York</code>
            </div>
          </div>

          {/* K-means */}
          <div
            style={{
              border: "1px solid rgba(255,255,255,0.12)",
              borderRadius: 16,
              padding: 14,
              background: "rgba(255,255,255,0.03)",
            }}
          >
            <div style={{ fontWeight: 800, marginBottom: 8 }}>K-Means Baseline</div>
            <div style={{ opacity: 0.78, lineHeight: 1.6, marginBottom: 10 }}>
              Compact districts from clustering precinct centroids (before optimization).
            </div>
            <img
              src="/about/kmeans_map.png"
              alt="K-means baseline districts"
              style={{
                width: "100%",
                height: "auto",
                objectFit: "cover",
                borderRadius: 12,
                border: "1px solid rgba(255,255,255,0.12)",
              }}
              onError={(e) => {
                (e.currentTarget as HTMLImageElement).style.display = "none";
              }}
            />
            <div style={{ marginTop: 10, opacity: 0.7, fontSize: 13 }}>
              <code>Our K-means Generated map of New York</code>
            </div>
          </div>
        </div>
      </section>

      {/* ---------- SECTION: HILLCLIMB WITH IMAGE(S) ---------- */}
      <section style={{ marginBottom: 26 }}>
        <h2 style={{ margin: "18px 0 10px", fontSize: 22 }}>Hill-Climb Optimization</h2>

        <div style={{ display: "grid", gridTemplateColumns: "1.1fr 0.9fr", gap: 16 }}>
          <div
            style={{
              border: "1px solid rgba(255,255,255,0.12)",
              borderRadius: 16,
              padding: 16,
              background: "rgba(255,255,255,0.03)",
            }}
          >
            <p style={{ marginTop: 0, opacity: 0.86, lineHeight: 1.75 }}>
              Hill climbing is a local search method: start from an initial plan (usually the K-Means output) and
              repeatedly propose small boundary changes that improve an objective.
            </p>

            <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.7, opacity: 0.88 }}>
              <li>
                <b>Move type:</b> swap a precinct (or small set) across a shared border.
              </li>
              <li>
                <b>Constraints:</b> maintain contiguity and keep district populations within a tolerance band.
              </li>
              <li>
                <b>Goal:</b> improve a score (e.g., expected seats for a target party) without breaking the map.
              </li>
              <li>
                <b>Stops when:</b> no proposed move yields improvement (a local optimum).
              </li>
            </ul>

            <div style={{ marginTop: 12, opacity: 0.75, lineHeight: 1.6 }}>
              In the flipbook, each frame is a snapshot of the map after a set of accepted moves, making it easier to
              see how the algorithm evolves districts over time.
            </div>
          </div>

          {/* Image grid for hillclimb frames */}
          <div
            style={{
              border: "1px dashed rgba(255,255,255,0.22)",
              borderRadius: 16,
              padding: 16,
              background: "rgba(255,255,255,0.02)",
            }}
          >
            <div style={{ fontWeight: 800, marginBottom: 8 }}>Hillclimb frames</div>

            <div style={{ display: "flex", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              {[
                { src: "/about/hillclimb_1.png", label: "Early" },
                { src: "/about/hillclimb_2.png", label: "Mid" },
                { src: "/about/hillclimb_3.png", label: "Late" },
             
              ].map((im) => (
                <div key={im.src} style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <img
                    src={im.src}
                    alt={`Hillclimb ${im.label}`}
                    style={{
                      width: "100%",
                      height: "auto",
                      objectFit: "cover",
                      borderRadius: 12,
                      border: "1px solid rgba(255,255,255,0.12)",
                    }}
                    onError={(e) => {
                      (e.currentTarget as HTMLImageElement).style.display = "none";
                    }}
                  />
                  <div style={{ fontSize: 12, opacity: 0.72 }}>{im.label}</div>
                </div>
              ))}
            </div>

            <div style={{ marginTop: 10, opacity: 0.7, fontSize: 13 }}>
              Placeholders: <code>/public/about/hillclimb_1.png</code> … <code>hillclimb_4.png</code>
            </div>
          </div>
        </div>
      </section>

      {/* ---------- SECTION: OUTPUTS ---------- */}
      <section style={{ marginBottom: 40 }}>
        <h2 style={{ margin: "18px 0 10px", fontSize: 22 }}>How outputs are served</h2>

        <div
          style={{
            border: "1px solid rgba(255,255,255,0.12)",
            borderRadius: 16,
            padding: 16,
            background: "rgba(255,255,255,0.03)",
          }}
        >
          <p style={{ marginTop: 0, opacity: 0.85, lineHeight: 1.75 }}>
            Offline scripts export each run into <code>/public/outputs</code>. The site lists available states and runs,
            and loads the selected GeoJSON + stats directly in the browser.
          </p>

          <div style={{ opacity: 0.82, lineHeight: 1.7 }}>
            Typical exported artifacts include:
            <ul style={{ marginTop: 8, paddingLeft: 18, lineHeight: 1.7 }}>
              <li>
                <code>map_data.geojson</code> — district geometry + properties for rendering
              </li>
              <li>
                <code>district_stats.json</code> (or similar) — per-district summary metrics
              </li>
              <li>
                <code>unit_to_district.csv</code> — assignment of each precinct/unit to a district label
              </li>
              <li>
                optional flipbook frames — images per step for “View Redistricting In Action”
              </li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}