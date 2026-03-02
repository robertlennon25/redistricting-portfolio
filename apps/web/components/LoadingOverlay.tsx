import React from "react";

export default function LoadingOverlay({ show, label }: { show: boolean; label?: string }) {
  if (!show) return null;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9999,
        display: "grid",
        placeItems: "center",
        background: "rgba(255,255,255,0.78)",
        backdropFilter: "blur(6px)",
      }}
    >
      <div style={{ textAlign: "center" }}>
        {/* "Globe" spinner */}
        <div
          style={{
            width: 84,
            height: 84,
            borderRadius: "50%",
            border: "6px solid rgba(0,0,0,0.10)",
            borderTopColor: "rgba(0,0,0,0.55)",
            margin: "0 auto",
            animation: "spin 0.9s linear infinite",
          }}
        />
        <div style={{ marginTop: 14, fontWeight: 900, letterSpacing: 0.2 }}>
          {label ?? "Loading run…"}
        </div>
        <div style={{ marginTop: 6, opacity: 0.7, fontSize: 13 }}>
          Fetching map + stats…
        </div>

        {/* Keyframes */}
        <style jsx>{`
          @keyframes spin {
            from {
              transform: rotate(0deg);
            }
            to {
              transform: rotate(360deg);
            }
          }
        `}</style>
      </div>
    </div>
  );
}