import { ReactNode, useState } from "react";
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ApiError } from "./api";
import { FeatureSpec, Patient, SchemaResp, useApp } from "./context";

export const C = { teal: "#3fc1b0", amber: "#e3a553", red: "#e57583", muted: "#92a6c2", faint: "#647a9b", line: "#223550", text: "#e5edf7", blue: "#7aa7e8", violet: "#a88be0" };
export const MODEL_COLORS: Record<string, string> = { logistic_regression: C.teal, random_forest: C.amber, svm_rbf: C.blue, gradient_boosting: C.violet, baseline: C.faint };
export const tooltipStyle = { contentStyle: { background: "#15243d", border: "1px solid #223550", borderRadius: 8, color: "#e5edf7", fontSize: 13 }, labelStyle: { color: "#92a6c2" }, itemStyle: { color: "#e5edf7" } };
export const axisProps = { stroke: C.faint, tick: { fill: C.muted, fontSize: 12 }, tickLine: false } as const;

export function Panel({ title, sub, aside, children, className = "" }: { title?: ReactNode; sub?: ReactNode; aside?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`panel ${className}`}>
      {(title || aside) && (
        <div className="panel-head"><div><h2>{title}</h2>{sub && <div className="sub">{sub}</div>}</div>{aside}</div>
      )}
      {children}
    </section>
  );
}
export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return <div className="stat"><div className="v">{value}</div><div className="l">{label}</div>{hint && <div className="h">{hint}</div>}</div>;
}
export function Loading({ label = "Loading from the backend…" }: { label?: string }) {
  return <div className="loading" role="status"><span className="spin" aria-hidden /> {label}</div>;
}
export function ErrorNote({ error, retry }: { error: ApiError | Error | null; retry?: () => void }) {
  if (!error) return null;
  const code = (error as ApiError).code;
  const needsTraining = code === "model_not_trained";
  return (
    <div className="note err" role="alert">
      <strong>{needsTraining ? "No trained model yet." : "Something went wrong."}</strong> {error.message}
      {needsTraining && <> Open <a href="/benchmark">ML benchmarking</a> and start a training run.</>}
      {retry && <> <button className="btn sm" onClick={retry}>Try again</button></>}
    </div>
  );
}
export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return <div className="empty"><h3>{title}</h3><div>{children}</div></div>;
}
export function Tabs<T extends string>({ tabs, value, onChange }: { tabs: { id: T; label: string }[]; value: T; onChange: (t: T) => void }) {
  return <div className="tabs" role="tablist">{tabs.map((t) => (
    <button key={t.id} role="tab" aria-selected={value === t.id} className={value === t.id ? "on" : ""} onClick={() => onChange(t.id)}>{t.label}</button>))}</div>;
}
export function ClassPill({ c }: { c: number }) {
  return <span className={`pill c${c}`}>target = {c}</span>;
}
export function Disclaimer({ children }: { children?: ReactNode }) {
  return <div className="note warn" role="note"><strong>Model output, not a diagnosis.</strong> {children ?? "CardioSage reports what a statistical model fitted to a small historical research dataset outputs. It is not medical advice, not clinically validated, and its probabilities are not calibrated patient risks."}</div>;
}
export function ModelPicker() {
  const { models, algorithm, setAlgorithm } = useApp();
  if (!models.length) return null;
  return (
    <div style={{ minWidth: 220 }}>
      <label className="f" htmlFor="algo">Model</label>
      <select id="algo" value={algorithm ?? ""} onChange={(e) => setAlgorithm(e.target.value || null)}>
        <option value="">Selected model ({models.find((m) => m.selected)?.label})</option>
        {models.filter((m) => m.key !== "baseline").map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
      </select>
    </div>
  );
}

/** The signature element: a probability band from class 0 (amber) to class 1 (teal) with the 0.5 decision threshold. */
export function DecisionStrip({ p1, pB }: { p1: number; pB?: number }) {
  const label = pB != null ? `Scenario A ${(p1 * 100).toFixed(0)}%, scenario B ${(pB * 100).toFixed(0)}% probability of target = 1` : `${(p1 * 100).toFixed(0)}% probability of target = 1`;
  return (
    <div className="strip" role="img" aria-label={label}>
      <span className="tick" />
      <span className={`mark ${pB != null ? "a" : ""}`} style={{ left: `${p1 * 100}%` }} />
      {pB != null && <span className="mark b" style={{ left: `${pB * 100}%` }} />}
      <span className="lab l">target = 0</span><span className="lab m">threshold 0.5</span><span className="lab r">target = 1</span>
    </div>
  );
}

export function ContribChart({ items, height }: { items: { label: string; value_display?: string; contribution: number }[]; height?: number }) {
  const data = items.map((i) => ({ name: i.value_display ? `${i.label}: ${i.value_display}` : i.label, v: i.contribution }));
  const m = Math.max(0.05, ...data.map((d) => Math.abs(d.v)));
  return (
    <ResponsiveContainer width="100%" height={height ?? Math.max(180, data.length * 34 + 30)}>
      <BarChart data={data} layout="vertical" margin={{ left: 10, right: 24, top: 4, bottom: 4 }}>
        <XAxis type="number" domain={[-m * 1.1, m * 1.1]} {...axisProps} tickFormatter={(v) => (v > 0 ? "+" : "") + v.toFixed(2)} />
        <YAxis type="category" dataKey="name" width={230} {...axisProps} tick={{ fill: C.text, fontSize: 12.5 }} />
        <ReferenceLine x={0} stroke={C.faint} />
        <Tooltip {...tooltipStyle} cursor={{ fill: "rgba(255,255,255,.04)" }} formatter={(v: number) => [(v > 0 ? "+" : "") + v.toFixed(3) + " probability", "Contribution"]} />
        <Bar dataKey="v" radius={3}>{data.map((d, i) => <Cell key={i} fill={d.v >= 0 ? C.teal : C.amber} />)}</Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function validatePatient(p: Patient, schema: SchemaResp): Record<string, string> {
  const errs: Record<string, string> = {};
  for (const f of schema.features) {
    const v = p[f.name];
    if (v === undefined || v === null || Number.isNaN(v)) { errs[f.name] = "Required"; continue; }
    if (f.hard && (v < f.hard[0] || v > f.hard[1])) errs[f.name] = `Must be between ${f.hard[0]} and ${f.hard[1]}${f.unit ? " " + f.unit : ""}`;
    else if (f.step === 1 && !Number.isInteger(v)) errs[f.name] = "Must be a whole number";
    else if (f.options && !(String(v) in f.options)) errs[f.name] = `Choose one of: ${Object.keys(f.options).join(", ")}`;
  }
  return errs;
}

function Field({ f, value, onChange, error, changed, observed, disabled }: { f: FeatureSpec; value: number; onChange: (v: number) => void; error?: string; changed?: boolean; observed?: any; disabled?: boolean }) {
  const [draft, setDraft] = useState<string | null>(null);
  const id = `f-${f.name}`;
  const cls = `field ${error ? "err" : ""} ${changed ? "changed" : ""}`;
  const labelEl = <label className="f" htmlFor={id}>{f.label}{f.unit ? ` (${f.unit})` : ""}</label>;
  if (f.options) {
    return (
      <div className={cls}>{labelEl}
        <select id={id} value={String(value)} disabled={disabled} aria-invalid={!!error} onChange={(e) => onChange(Number(e.target.value))}>
          {Object.entries(f.options).map(([k, l]) => <option key={k} value={k}>{k} · {l}</option>)}
        </select>
        {error && <div className="msg">{error}</div>}
      </div>
    );
  }
  const lo = observed?.min ?? f.hard![0], hi = observed?.max ?? f.hard![1];
  const outside = Number.isFinite(value) && (value < lo || value > hi);
  return (
    <div className={cls}>{labelEl}
      <div className="range-row">
        <input type="range" aria-label={`${f.label} slider`} min={lo} max={hi} step={f.step ?? 1} value={Math.min(hi, Math.max(lo, Number.isFinite(value) ? value : lo))} disabled={disabled} onChange={(e) => { setDraft(null); onChange(Number(e.target.value)); }} />
        <input id={id} type="number" inputMode="decimal" step={f.step ?? 1} value={draft ?? (Number.isFinite(value) ? String(value) : "")} disabled={disabled} aria-invalid={!!error}
          onChange={(e) => { setDraft(e.target.value); const n = e.target.value === "" ? NaN : Number(e.target.value); onChange(n); }} onBlur={() => setDraft(null)} />
      </div>
      {error ? <div className="msg">{error}</div> : <div className="hint">{outside ? `Outside the training range (${lo}–${hi}): the model will extrapolate.` : `Training data range ${lo}–${hi}`}</div>}
    </div>
  );
}

export function PatientForm({ value, onChange, errors = {}, baseline, disabled, columns = 1 }: { value: Patient; onChange: (p: Patient) => void; errors?: Record<string, string>; baseline?: Patient; disabled?: boolean; columns?: 1 | 2 }) {
  const { schema, schemaError } = useApp();
  if (schemaError) return <ErrorNote error={schemaError} />;
  if (!schema) return <Loading label="Loading feature schema…" />;
  return (
    <div style={columns === 2 ? { display: "grid", gridTemplateColumns: "1fr 1fr", columnGap: 22 } : undefined}>
      {schema.features.map((f) => (
        <Field key={f.name} f={f} value={value[f.name]} disabled={disabled} observed={schema.observed_ranges[f.name]}
          error={errors[f.name]} changed={baseline ? baseline[f.name] !== value[f.name] : false}
          onChange={(v) => onChange({ ...value, [f.name]: v })} />
      ))}
    </div>
  );
}

export function Icon({ name }: { name: string }) {
  const p: Record<string, string> = {
    overview: "M3 12l9-8 9 8M5 10v10h5v-6h4v6h5V10",
    data: "M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zm0 0v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6",
    predict: "M3 12h4l3-8 4 16 3-8h4",
    cf: "M4 7h11l-3-3M20 17H9l3 3M4 7v0M20 17v0",
    whatif: "M12 3v18M3 12h18M7 7l10 10",
    bench: "M5 20V10M12 20V4M19 20v-7",
    explain: "M12 3a6 6 0 00-4 10.5V17h8v-3.5A6 6 0 0012 3zM9 21h6",
    history: "M12 7v5l3 2M3 12a9 9 0 109-9 9 9 0 00-7 3.4M3 4v4h4",
    method: "M6 3h9l4 4v14H6zM14 3v5h5M9 13h7M9 17h7",
  };
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d={p[name]} /></svg>;
}

export function PageHead({ title, children }: { title: string; children?: ReactNode }) {
  return <header className="page-head"><h1>{title}</h1>{children && <p>{children}</p>}</header>;
}
