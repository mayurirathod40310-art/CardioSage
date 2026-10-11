import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, pct, useAction } from "../api";
import { ClassPill, DecisionStrip, ErrorNote, ModelPicker, PageHead, Panel, PatientForm, validatePatient } from "../components";
import { useApp } from "../context";

const show = (c: any) => c.kind === "numeric"
  ? <><span className="from num">{c.from}</span> → <span className="to num">{c.to}</span>{c.unit ? ` ${c.unit}` : ""}</>
  : <><span className="from">{c.from_display ?? c.from}</span> → <span className="to">{c.to_display ?? c.to}</span></>;

export default function Counterfactual() {
  const { patient, setPatient, schema, algorithm, setScenarioB } = useApp();
  const nav = useNavigate();
  const [locked, setLocked] = useState<string[]>(["age", "sex"]);
  const [maxChanges, setMax] = useState(3);
  const [margin, setMargin] = useState(0.05);
  const [n, setN] = useState(5);
  const [desired, setDesired] = useState<"" | "0" | "1">("");
  const [localErr, setLocalErr] = useState<string | null>(null);
  const act = useAction((body: any) => api("/counterfactual", { body }));
  const r = act.data;

  const run = () => {
    if (!schema) return;
    const e = validatePatient(patient, schema);
    if (Object.keys(e).length) { setLocalErr("Fix the highlighted inputs first: " + Object.keys(e).join(", ")); return; }
    setLocalErr(null);
    act.run({ features: patient, algorithm, desired_class: desired === "" ? null : Number(desired), locked_features: locked, max_changes: maxChanges, n_results: n, margin });
  };
  const toggle = (f: string) => setLocked((l) => (l.includes(f) ? l.filter((x) => x !== f) : [...l, f]));

  return (
    <>
      <PageHead title="Counterfactual analysis">Which small input changes flip the model’s predicted class? Every candidate below was re-scored with the saved model before being shown.</PageHead>
      <div className="note warn" style={{ marginBottom: 18 }}><strong>Model counterfactuals, not medical recommendations.</strong> They describe the fitted function. They do not show that changing a real patient’s measurement would change their condition, and some combinations may never occur in real people.</div>
      <div className="split">
        <div className="stack">
          <Panel title="Record">
            <details><summary style={{ cursor: "pointer", color: "var(--teal)", marginBottom: 12 }}>Edit the record ({Object.keys(patient).length} fields)</summary>
              <PatientForm value={patient} onChange={setPatient} /></details>
            {!schema ? null : <p className="small muted" style={{ margin: 0 }}>Same record as the prediction studio: age {patient.age}, {schema.features.find((f) => f.name === "sex")!.options![String(patient.sex)]?.toLowerCase()}, max heart rate {patient.thalach}, ST depression {patient.oldpeak}.</p>}
          </Panel>
          <Panel title="Search settings">
            <div className="field"><ModelPicker /></div>
            <div className="field"><label className="f" htmlFor="des">Desired outcome</label>
              <select id="des" value={desired} onChange={(e) => setDesired(e.target.value as any)}><option value="">Opposite of the current prediction</option><option value="0">target = 0</option><option value="1">target = 1</option></select></div>
            <div className="field"><label className="f">Maximum features changed</label>
              <div className="seg">{[1, 2, 3].map((k) => <button key={k} className={maxChanges === k ? "on" : ""} onClick={() => setMax(k)}>{k}</button>)}</div></div>
            <div className="field"><label className="f" htmlFor="mg">Required probability margin beyond 0.5: {margin.toFixed(2)}</label>
              <input id="mg" type="range" min={0} max={0.3} step={0.01} value={margin} onChange={(e) => setMargin(Number(e.target.value))} /><div className="hint">A margin avoids borderline flips that a tiny change could undo.</div></div>
            <div className="field"><label className="f" htmlFor="nr">Results to show</label>
              <select id="nr" value={n} onChange={(e) => setN(Number(e.target.value))}>{[3, 5, 8].map((k) => <option key={k}>{k}</option>)}</select></div>
            <fieldset style={{ border: 0, padding: 0, margin: "0 0 14px" }}><legend className="f" style={{ marginBottom: 8 }}>Keep these fixed</legend>
              <div className="grid g2" style={{ gap: 6 }}>{schema?.features.map((f) => (
                <label key={f.name} className="row" style={{ gap: 8, fontSize: ".88rem", color: "var(--text)" }}><input type="checkbox" checked={locked.includes(f.name)} onChange={() => toggle(f.name)} />{f.label}</label>))}</div></fieldset>
            <button className="btn primary" onClick={run} disabled={act.loading}>{act.loading ? <><span className="spin" /> Searching…</> : "Find counterfactuals"}</button>
            {localErr && <div className="note err" style={{ marginTop: 12 }}>{localErr}</div>}
            {act.error && <div style={{ marginTop: 12 }}><ErrorNote error={act.error} /></div>}
          </Panel>
        </div>

        <div>
          {!r && !act.loading && <div className="empty"><h3>No search yet</h3>Choose which features may change and run the search. Fixed features are never altered, and changes stay inside the ranges seen in the data.</div>}
          {act.loading && <div className="panel"><span className="spin" /> Evaluating thousands of candidate inputs with the saved model…</div>}
          {r && (
            <div className="stack">
              <Panel title="Starting point">
                <div className="row">Model output <ClassPill c={r.original.predicted_class} /> · P(target = 1) {pct(r.original.probability_class1)} · goal <ClassPill c={r.desired_class} /> with probability ≥ {r.required_probability.toFixed(2)}</div>
                <DecisionStrip p1={r.original.probability_class1} />
                <div className="faint small">{r.evaluated.toLocaleString()} candidates evaluated in {r.seconds}s using {r.algorithm} · {r.free_features.length} features free to change · saved as experiment #{r.experiment_id}</div>
              </Panel>
              {!r.found && <div className="note high"><strong>No valid counterfactual found.</strong> {r.reason}{r.best_attempt_changes && <> Closest attempt changed {Object.entries(r.best_attempt_changes).map(([k, v]) => `${k} → ${v}`).join(", ")}.</>}</div>}
              {r.found && r.candidates.map((c: any, i: number) => (
                <div key={i} className="cf-card">
                  <div className="row spread"><h3>Option {i + 1}: change {c.n_changes} feature{c.n_changes > 1 ? "s" : ""}</h3>
                    <div className="row"><span className="pill c1">re-verified</span><span className={`pill ${c.typical ? "neutral" : "bad"}`}>{c.typical ? "typical" : "unusual combination"}</span></div></div>
                  {c.changes.map((x: any) => <div key={x.feature} className="chg"><b>{x.label}</b><span>{show(x)}</span>{x.kind === "numeric" && <span className="d">{x.delta > 0 ? "+" : ""}{x.delta}{x.unit ? ` ${x.unit}` : ""} · {pct(x.relative_to_range, 0)} of the observed range</span>}</div>)}
                  <DecisionStrip p1={r.original.probability_class1} pB={c.probability_class1} />
                  <div className="row spread small muted"><span>New model output <ClassPill c={c.predicted_class} /> at {pct(c.probability_class1)} probability of target = 1</span>
                    <button className="btn sm" onClick={() => { setScenarioB(c.input); nav("/what-if"); }}>Open in what-if lab</button></div>
                  <div className="faint small" style={{ marginTop: 6 }}>{c.plausibility_note}.</div>
                </div>))}
              <div className="faint small">{r.disclaimer}</div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
