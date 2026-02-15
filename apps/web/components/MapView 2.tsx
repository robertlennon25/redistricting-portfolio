import React, { useEffect, useMemo, useState } from "react";
import L from "leaflet";
import { MapContainer, TileLayer, GeoJSON, useMap } from "react-leaflet";
import type { Feature, FeatureCollection, Geometry } from "geojson";
import type { DistrictStat } from "./Sidebar";

function colorForDistrict(d: number) {
  const hue = (d * 47) % 360;
  return `hsl(${hue} 70% 55%)`;
}

function fmt(n: any) {
  const x = Number(n);
  if (!Number.isFinite(x)) return "";
  return Math.round(x).toLocaleString();
}

function FitToGeoJSON({ geojson }: { geojson: FeatureCollection }) {
  const map = useMap();

  useEffect(() => {
    try {
      const layer = L.geoJSON(geojson as any);
      const bounds = layer.getBounds();
      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [20, 20] });
      }
    } catch (e) {
      console.error("fitBounds failed:", e);
    }
  }, [geojson, map]);

  return null;
}

export default function MapView({
  geojson,
  statsByDistrict,
  hoverDistrict,
  setHoverDistrict
}: {
  geojson: FeatureCollection;
  statsByDistrict: Map<number, DistrictStat>;
  hoverDistrict: number | null;
  setHoverDistrict: (d: number | null) => void;
}) {
  const styleFn = (feature?: Feature<Geometry, any>) => {
    const d = Number(feature?.properties?.district ?? -1);
    const isHover = hoverDistrict !== null && d === hoverDistrict;
    return {
      color: isHover ? "#000" : "#ffffff",
      weight: isHover ? 1.6 : 0.2,
      fillOpacity: 0.65,
      fillColor: d >= 0 ? colorForDistrict(d) : "#999999"
    };
  };

  const onEachFeature = (feature: Feature<Geometry, any>, layer: any) => {
    const props = feature.properties || {};
    const d = Number(props.district);
    const districtStats = statsByDistrict.get(d);

    const html = `
      <div style="font-family: ui-sans-serif, system-ui; font-size: 12px;">
        <div style="font-weight: 700; margin-bottom: 4px;">District ${d}</div>
        <div><b>Precinct</b></div>
        <div>Dem: ${fmt(props.dem_votes)} | GOP: ${fmt(props.rep_votes)} | Pop: ${fmt(props.weight)}</div>
        <div style="margin-top: 6px;"><b>District totals</b></div>
        <div>Dem: ${fmt(districtStats?.dem_votes)} | GOP: ${fmt(districtStats?.rep_votes)} | Pop: ${fmt(districtStats?.weight)}</div>
        <div>Winner: ${districtStats?.winner ?? ""} | Margin: ${fmt(districtStats?.margin)} (${Number(districtStats?.margin_pct ?? 0).toFixed(1)}%)</div>
      </div>
    `;

    layer.bindTooltip(html, { sticky: true });
    layer.on({
      mouseover: () => setHoverDistrict(d),
      mouseout: () => setHoverDistrict(null)
    });
  };

  // IMPORTANT: for performance with big GeoJSON, force Canvas renderer
  const canvas = useMemo(() => L.canvas({ padding: 0.5 }), []);

  return (
    <MapContainer
      center={[40.0, -89.0]}
      zoom={6}
      style={{ height: "100vh", width: "100%" }}
      preferCanvas={true}
    >
      <TileLayer
        attribution='&copy; OpenStreetMap'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      <FitToGeoJSON geojson={geojson} />

      <GeoJSON
        key={(geojson as any)?.features?.length ?? 0}   // forces refresh if data changes
        data={geojson as any}
        style={styleFn as any}
        onEachFeature={onEachFeature as any}
        renderer={canvas as any}
      />
    </MapContainer>
  );
}
