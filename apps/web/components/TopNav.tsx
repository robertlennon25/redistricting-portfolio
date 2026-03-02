import Link from "next/link";
import { useRouter } from "next/router";

type NavItem = { href: string; label: string };

const items: NavItem[] = [
  { href: "/", label: "Map" },
  { href: "/in-action", label: "View Redistricting In Action" },
  { href: "/about", label: "About" },
];

export default function TopNav() {
  const router = useRouter();

  return (
    <div
      style={{
        position: "sticky",
        top: 0,
        zIndex: 50,
        background: "white",
        borderBottom: "1px solid rgba(0,0,0,0.08)",
      }}
    >
      <div
        style={{
          maxWidth: 1200,
          margin: "0 auto",
          padding: "10px 16px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ fontWeight: 900, letterSpacing: 0.2 }}>Gerrymandering Demo</div>
          <div style={{ opacity: 0.35 }}>|</div>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            {items.map((it) => {
              const active = router.pathname === it.href;
              return (
                <Link
                  key={it.href}
                  href={it.href}
                  style={{
                    padding: "8px 10px",
                    borderRadius: 10,
                    textDecoration: "none",
                    color: "black",
                    border: active ? "1px solid rgba(0,0,0,0.18)" : "1px solid transparent",
                    background: active ? "rgba(0,0,0,0.04)" : "transparent",
                    fontWeight: active ? 800 : 600,
                  }}
                >
                  {it.label}
                </Link>
              );
            })}
          </div>
        </div>

        <div style={{ fontSize: 12, opacity: 0.7 }}>
          {/* space for later (e.g. version, GitHub link) */}
        </div>
      </div>
    </div>
  );
}