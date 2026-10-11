import { useEffect, useState, type FormEvent } from "react";
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

type AuthUser = { id: number; email: string; role: string };
type AuthMode = "login" | "register";

export default function App() {
  const { data: h } = useApi(() => api("/health"), []);
  const [user, setUser] = useState<AuthUser | null>(null);
  const [sessionChecked, setSessionChecked] = useState(false);
  const [authOpen, setAuthOpen] = useState(false);
  const [authMode, setAuthMode] = useState<AuthMode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [authError, setAuthError] = useState("");
  const [authBusy, setAuthBusy] = useState(false);

  useEffect(() => {
    let active = true;
    api<AuthUser>("/auth/me")
      .then((me) => { if (active) setUser(me); })
      .catch(() => { if (active) setUser(null); })
      .finally(() => { if (active) setSessionChecked(true); });
    return () => { active = false; };
  }, []);

  function openAuth(mode: AuthMode = "login") {
    setAuthMode(mode);
    setAuthError("");
    setAuthOpen(true);
  }

  async function submitAuth(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setAuthBusy(true);
    setAuthError("");
    try {
      const credentials = { email: email.trim(), password };
      if (authMode === "register") {
        await api("/auth/register", { body: credentials });
      }
      await api("/auth/login", { body: credentials });
      const me = await api<AuthUser>("/auth/me");
      setUser(me);
      setPassword("");
      setAuthOpen(false);
    } catch (error) {
      setAuthError(error instanceof Error ? error.message : "Sign-in failed. Please try again.");
    } finally {
      setAuthBusy(false);
    }
  }

  async function signOut() {
    try {
      await api("/auth/logout", { method: "POST" });
      setUser(null);
      setPassword("");
      setAuthError("");
      setAuthOpen(false);
      // Clear locally persisted feature inputs when leaving an account.
      localStorage.removeItem("cs.patient");
      localStorage.removeItem("cs.scenarioB");
      window.location.reload();
    } catch (error) {
      setAuthError(error instanceof Error ? error.message : "Could not sign out.");
      setAuthOpen(true);
    }
  }

  return (
    <div className="shell">
      <aside className="side">
        <div className="brand"><Logo /><b>CardioSage</b></div>
        <nav className="nav" aria-label="Main">
          {NAV.map((n) => <NavLink key={n.to} to={n.to} end={n.end}><Icon name={n.icon} />{n.label}</NavLink>)}
        </nav>
        <div className="side-foot">
          <div><span className={`pill ${h?.status === "ok" ? "c1" : "neutral"}`}>{h ? (h.status === "ok" ? "Backend online" : "Backend degraded") : "Connecting?"}</span></div>
          {h?.active_model_version && <div style={{ marginTop: 8 }}>Model {h.active_model_version}</div>}
          <div style={{ marginTop: 8 }}>Research software. Not a medical device.</div>
        </div>
      </aside>
      <div>
        <nav className="mobile-nav" aria-label="Main (compact)">
          {NAV.map((n) => <NavLink key={n.to} to={n.to} end={n.end}>{n.label}</NavLink>)}
        </nav>
        <main className="main">
          <div className="account-strip">
            {!sessionChecked ? <span className="account-status">Checking session?</span> :
              user ? <span className="account-status">Signed in as <b>{user.email}</b></span> :
              <span className="account-status">Guest mode: calculations work, but results are not saved to history.</span>}
            {sessionChecked && (user ?
              <button className="account-button secondary" type="button" onClick={() => { void signOut(); }}>Sign out</button> :
              <button className="account-button" type="button" onClick={() => openAuth("login")}>Sign in / Register</button>)}
          </div>

          {authOpen && !user && (
            <div className="auth-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) setAuthOpen(false); }}>
              <section className="auth-card" role="dialog" aria-modal="true" aria-labelledby="auth-title">
                <div className="auth-title-row">
                  <div>
                    <p className="auth-eyebrow">CARDIOSAGE ACCOUNT</p>
                    <h2 id="auth-title">{authMode === "login" ? "Welcome back" : "Create your account"}</h2>
                  </div>
                  <button className="auth-close" type="button" onClick={() => setAuthOpen(false)} aria-label="Close">?</button>
                </div>
                <p className="auth-description">Sign in to keep your own prediction and experiment history private to your account.</p>
                <form onSubmit={(event) => { void submitAuth(event); }}>
                  <label className="auth-field">
                    <span>Email address</span>
                    <input type="email" value={email} onChange={(event) => setEmail(event.target.value)}
                      autoComplete="email" required maxLength={254} placeholder="you@example.com" />
                  </label>
                  <label className="auth-field">
                    <span>Password{authMode === "register" ? " (12 characters minimum)" : ""}</span>
                    <input type="password" value={password} onChange={(event) => setPassword(event.target.value)}
                      autoComplete={authMode === "register" ? "new-password" : "current-password"}
                      minLength={authMode === "register" ? 12 : 1} maxLength={128} required placeholder="Enter your password" />
                  </label>
                  {authError && <p className="auth-error" role="alert">{authError}</p>}
                  <button className="account-button auth-submit" type="submit" disabled={authBusy}>
                    {authBusy ? "Please wait?" : authMode === "login" ? "Sign in" : "Create account"}
                  </button>
                </form>
                <p className="auth-switch">
                  {authMode === "login" ? "New to CardioSage?" : "Already have an account?"}{" "}
                  <button type="button" onClick={() => { setAuthMode(authMode === "login" ? "register" : "login"); setAuthError(""); }}>
                    {authMode === "login" ? "Create account" : "Sign in"}
                  </button>
                </p>
                <p className="auth-footnote">CardioSage is research software, not a medical device or a diagnosis.</p>
              </section>
            </div>
          )}

          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/dataset" element={<Dataset />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/counterfactual" element={<Counterfactual />} />
            <Route path="/what-if" element={<WhatIf />} />
            <Route path="/benchmark" element={<Benchmark />} />
            <Route path="/explain" element={<Explain />} />
            <Route path="/history" element={
              !sessionChecked ? <div className="guest-history"><h2>Checking account?</h2></div> :
              user ? <History /> :
              <div className="guest-history">
                <h2>Your history is private</h2>
                <p>Sign in to view or delete the predictions and experiments saved to your account.</p>
                <button className="account-button" type="button" onClick={() => openAuth("login")}>Sign in to view history</button>
              </div>
            } />
            <Route path="/methodology" element={<Methodology />} />
            <Route path="*" element={<div className="empty"><h3>Page not found</h3><a href="/">Back to overview</a></div>} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
