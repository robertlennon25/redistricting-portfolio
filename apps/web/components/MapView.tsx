import React, { useEffect, useMemo } from "react";
import L from "leaflet";
import { MapContainer, TileLayer, GeoJSON, useMap } from "react-leaflet";
import type { Feature, FeatureCollection, Geometry } from "geojson";
import type { DistrictStat } from "./Sidebar";



function ForceCanvasRenderer() {
  const map = useMap();

  useEffect(() => {
    // Force vector layers to render with canvas
    // (leaflet typings don't always expose this nicely)
    // @ts-ignore
    map.options.renderer = L.canvas();
  }, [map]);

  return null;
}

type ColorMode = "rainbow" | "party";

function colorForDistrict(d: number) {
  const hue = (d * 47) % 360;
  return `hsl(${hue} 70% 55%)`;
}

function colorForWinner(w: string | undefined) {
  if (!w) return "#999999";
  const ww = String(w).toLowerCase();
  if (ww.includes("dem")) return "#2b6fff";
  if (ww.includes("gop") || ww.includes("rep")) return "#ff3b3b";
  return "#999999";
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
  districtsGeojson,
  statsByDistrict,
  hoverDistrict,
  setHoverDistrict,
  colorMode,
  showDistrictOutlines,
  outlineWeight
}: {
  geojson: FeatureCollection;
  districtsGeojson: FeatureCollection | null;
  statsByDistrict: Map<number, DistrictStat>;
  hoverDistrict: number | null;
  setHoverDistrict: (d: number | null) => void;
  colorMode: ColorMode;
  showDistrictOutlines: boolean;
  outlineWeight: number;
}) {
  const styleFn = (feature?: Feature<Geometry, any>) => {
    const d = Number(feature?.properties?.district ?? -1);
    const isHover = hoverDistrict !== null && d === hoverDistrict;

    const winner = feature?.properties?.district_winner as string | undefined;

    const fillColor =
      colorMode === "party"
        ? colorForWinner(winner)
        : d >= 0
          ? colorForDistrict(d)
          : "#999999";

    return {
      // precinct border styling (subtle)
      color: isHover ? "#000" : "#ffffff",
      weight: isHover ? 1.6 : 0.2,
      opacity: isHover ? 0.9 : 0.4,
      fillOpacity: 0.65,
      fillColor
    };
  };

  const districtOutlineStyle = (feature?: Feature<Geometry, any>) => {
    const d = Number(feature?.properties?.district ?? -1);
    const isHover = hoverDistrict !== null && d === hoverDistrict;

    return {
      color: "#000000",
      weight: isHover ? outlineWeight + 1.5 : outlineWeight,
      opacity: isHover ? 0.9 : 0.65,
      fillOpacity: 0.0
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

  // const canvas = useMemo(() => L.canvas({ padding: 0.5 }), []);
  const geoSig = useMemo(() => {
    // changes when the run changes; avoids expensive hashing
    const first = (geojson as any)?.features?.[0]?.properties?.unit_id ?? "";
    const last = (geojson as any)?.features?.[(geojson as any)?.features?.length - 1]?.properties?.unit_id ?? "";
    return `${(geojson as any)?.features?.length ?? 0}_${first}_${last}`;
  }, [geojson]);

  const statsSig = useMemo(() => {
    // changes when stats map is (re)created
    return String(statsByDistrict?.size ?? 0);
  }, [statsByDistrict]);

  return (
  <MapContainer
    center={[40.0, -89.0]}
    zoom={6}
    style={{ height: "calc(100vh - 56px)", width: "100%" }}
    preferCanvas={true}
  >
    <ForceCanvasRenderer />

    <TileLayer
      attribution='&copy; OpenStreetMap'
      url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
    />

    <FitToGeoJSON geojson={geojson} />

    {/* Precinct fill layer */}
    <GeoJSON
      key={`precincts_${geoSig}_${colorMode}_${statsSig}`}
      data={geojson as any}
      style={styleFn as any}
      onEachFeature={onEachFeature as any}
    />

    {/* District boundary overlay */}
    {showDistrictOutlines && districtsGeojson ? (
      <GeoJSON
        key={`districts_${(districtsGeojson as any)?.features?.length ?? 0}_${outlineWeight}`}
        data={districtsGeojson as any}
        style={districtOutlineStyle as any}
      />
    ) : null}
  </MapContainer>
  );
}
