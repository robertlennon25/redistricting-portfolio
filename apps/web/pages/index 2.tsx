import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import Sidebar, { DistrictStat } from "../components/Sidebar";

const MapView = dynamic(() => import("../components/MapView"), { ssr: false });

export default function Home() {
  const [geojson, setGeojson] = useState<any | null>(null);
  const [stats, setStats] = useState<DistrictStat[] | null>(null);
  const [hoverDistrict, setHoverDistrict] = useState<number | null>(null);

  useEffect(() => {
    (async () => {
      const [gj, st] = await Promise.all([
        fetch("/data/map_data.geojson").then(r => r.json()),
        fetch("/data/district_stats.json").then(r => r.json())
      ]);
      setGeojson(gj);
      setStats(st);
    })();
  }, []);

  const statsByDistrict = useMemo(() => {
    const m = new Map<number, DistrictStat>();
    (stats ?? []).forEach(s => m.set(Number(s.district), s));
    return m;
  }, [stats]);

  return (
    <div className="appShell">
      <div className="sidebar">
        <div className="headerRow">
          <h2 style={{ margin: 0 }}>Illinois Gerrymandering Demo</h2>
          <span className="badge">Greedy Packing</span>
        </div>
        <div className="small">
          Hover precincts to see precinct + district totals. Sidebar highlights hovered district.
        </div>

        <Sidebar
          stats={stats ?? []}
          activeDistrict={hoverDistrict}
          onHoverDistrict={setHoverDistrict}
        />
      </div>

      <div className="main">
        {!geojson || !stats ? (
          <div className="mapLoading">Loading map data…</div>
        ) : null}

        <MapView
          geojson={geojson}
          statsByDistrict={statsByDistrict}
          hoverDistrict={hoverDistrict}
          setHoverDistrict={setHoverDistrict}
        />
      </div>
    </div>
  );
}
