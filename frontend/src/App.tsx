import { NavLink, Route, Routes } from "react-router-dom";
import { Icon } from "./components";
import { useApi, api } from "./api";
import Overview from "./pages/Overview";
import Dataset from "./pages/Dataset";
import Predict from "./pages/Predict";
import Counterfactual from "./pages/Counterfactual";
import WhatIf from "./pages/WhatIf";
import Benchmark from "./pages/Benchmark";
import Explain from "./pages/Explain";
import History from "./pages/History";
import Methodology from "./pages/Methodology";

const NAV = [
  { to: "/", label: "Overview", icon: "overview", end: true },
  { to: "/dataset", label: "Dataset explorer", icon: "data" },
  { to: "/predict", label: "Prediction studio", icon: "predict" },
  { to: "/counterfactual", label: "Counterfactuals", icon: "cf" },
  { to: "/what-if", label: "What-if laboratory", icon: "whatif" },
  { to: "/benchmark", label: "ML benchmarking", icon: "bench" },
  { to: "/explain", label: "Explainability", icon: "explain" },
  { to: "/history", label: "History", icon: "history" },
  { to: "/methodology", label: "Methodology", icon: "method" },
];

function Logo() {
  return (
    <svg width="34" height="34" viewBox="0 0 64 64" aria-hidden>
      <rect width="64" height="64" rx="14" fill="#15243d" />
      <path d="M6 34h12l5-14 8 28 7-22 4 8h16" fill="none" stroke="#3fc1b0" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function App() {
  const { data: h } = useApi(() => api("/health"), []);
  return (
    <div className="shell">
      <aside className="side">
        <div className="brand"><Logo /><b>CardioSage</b></div>
        <nav className="nav" aria-label="Main">
          {NAV.map((n) => <NavLink key={n.to} to={n.to} end={n.end}><Icon name={n.icon} />{n.label}</NavLink>)}
        </nav>
        <div className="side-foot">
          <div><span className={`pill ${h?.status === "ok" ? "c1" : "neutral"}`}>{h ? (h.status === "ok" ? "Backend online" : "Backend degraded") : "Connecting…"}</span></div>
          {h?.active_model_version && <div style={{ marginTop: 8 }}>Model {h.active_model_version}</div>}
          <div style={{ marginTop: 8 }}>Research software. Not a medical device.</div>
        </div>
      </aside>
      <div>
        <nav className="mobile-nav" aria-label="Main (compact)">
          {NAV.map((n) => <NavLink key={n.to} to={n.to} end={n.end}>{n.label}</NavLink>)}
        </nav>
        <main className="main">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/dataset" element={<Dataset />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/counterfactual" element={<Counterfactual />} />
            <Route path="/what-if" element={<WhatIf />} />
            <Route path="/benchmark" element={<Benchmark />} />
            <Route path="/explain" element={<Explain />} />
            <Route path="/history" element={<History />} />
            <Route path="/methodology" element={<Methodology />} />
            <Route path="*" element={<div className="empty"><h3>Page not found</h3><a href="/">Back to overview</a></div>} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
