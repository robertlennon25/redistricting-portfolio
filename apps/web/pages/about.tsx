export default function AboutPage() {
  return (
    <div style={{ padding: 24, maxWidth: 980, margin: "0 auto" }}>
      <h1 style={{ marginTop: 8 }}>About</h1>
      <p style={{ opacity: 0.8, lineHeight: 1.55 }}>
        This demo visualizes redistricting runs generated offline (K-Means, hillclimb, and other algorithms)
        and served as static outputs to the web app.
      </p>

      <h3>What you’re looking at</h3>
      <ul style={{ lineHeight: 1.6 }}>
        <li><b>Map</b>: explore precomputed runs per state.</li>
        <li><b>View Redistricting In Action</b>: watch a flipbook of the algorithm’s steps.</li>
      </ul>

      <h3>How it works</h3>
      <p style={{ opacity: 0.8, lineHeight: 1.55 }}>
        The backend scripts export GeoJSON + stats into <code>/public/outputs</code>. The frontend simply fetches those
        files. No server required.
      </p>
    </div>
  );
}