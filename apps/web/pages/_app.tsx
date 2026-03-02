import type { AppProps } from "next/app";
import "leaflet/dist/leaflet.css";
import "../styles/globals.css";
import TopNav from "../components/TopNav";

export default function App({ Component, pageProps }: AppProps) {
  return (
    <div style={{ minHeight: "100vh" }}>
      <TopNav />
      <Component {...pageProps} />
    </div>
  );
}