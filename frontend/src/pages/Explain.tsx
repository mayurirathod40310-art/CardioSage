import { useState } from "react";
import { Link } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, ErrorBar, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, f3, pct, useAction, useApi } from "../api";
import { C, ClassPill, ContribChart, DecisionStrip, ErrorNote, Loading, ModelPicker, PageHead, Panel, Tabs, axisProps, tooltipStyle } from "../components";
import { useApp } from "../context";

type Tab = "global" | "local" | "concepts";
export default function Explain() {
  const { algorithm, patient } = useApp();
  const [tab, setTab] = useState<Tab>("global");
  const g = useApi(() => api(`/explain/global${algorithm ? `?algorithm=${algorithm}` : ""}`), [algorithm]);
  const loc = useAction((body: any) => api("/explain", { body }));
  const r = loc.data;
  return (
    <>
      <PageHead title="Explainability center">Three different questions get confused here: what the data shows, what the model relies on, and why it scored one patient the way it did. None of them is about cause.</PageHead>
      <Tabs value={tab} onChange={setTab} tabs={[{ id: "global", label: "What the model relies on" }, { id: "local", label: "Explain one prediction" }, { id: "concepts", label: "Concepts" }]} />
      {tab === "global" && (
        <div className="stack">
          <Panel title="Permutation importance" sub="Drop in held-out ROC-AUC when one feature’s values are shuffled (20 repeats). Bars show the mean; whiskers one standard deviation." aside={<ModelPicker />}>
            {g.loading && !g.data ? <Loading /> : g.error ? <ErrorNote error={g.error} retry={g.reload} /> : g.data && (
              <>
                <ResponsiveContainer width="100%" height={420}>
                  <BarChart data={g.data.permutation_importance} layout="vertical" margin={{ left: 10, right: 30 }}>
                    <CartesianGrid stroke={C.line} horizontal={false} /><XAxis type="number" {...axisProps} /><YAxis type="category" dataKey="feature" width={80} {...axisProps} tick={{ fill: C.text, fontSize: 12.5 }} />
                    <Tooltip {...tooltipStyle} cursor={{ fill: "rgba(255,255,255,.04)" }} formatter={(v: number) => v.toFixed(4)} />
                    <Bar dataKey="mean" fill={C.teal} radius={3}><ErrorBar dataKey="std" stroke={C.muted} width={4} /></Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div className="note">Importance is estimated on only {g.data.n_test} test records, so whiskers overlap for most features. Only the top few are reliably ahead of the rest. Importance says a model <em>uses</em> a feature, not that the feature <em>causes</em> disease.</div>
              </>)}
          </Panel>
          {g.data && <Panel title="Importance next to correlation" sub="They answer different questions and can disagree">
            <div className="scroll"><table className="tbl"><thead><tr><th>Feature</th><th className="r">Model importance</th><th className="r">± SD</th><th className="r">Correlation with target</th></tr></thead>
              <tbody>{[...g.data.comparison].sort((a: any, b: any) => b.permutation_importance - a.permutation_importance).map((x: any) => <tr key={x.feature}><td>{x.label}</td><td className="r">{x.permutation_importance.toFixed(4)}</td><td className="r">{x.permutation_std.toFixed(4)}</td><td className="r">{f3(x.pearson_with_target)}</td></tr>)}</tbody></table></div></Panel>}
        </div>
      )}
      {tab === "local" && (
        <div className="stack">
          <Panel title="Explain the current record" sub="Uses the record from the prediction studio. Change it there or load a real one." aside={<ModelPicker />}>
            <div className="row"><button className="btn primary" disabled={loc.loading} onClick={() => loc.run({ features: patient, algorithm })}>{loc.loading ? <><span className="spin" /> Explaining…</> : "Explain this prediction"}</button><Link className="btn" to="/predict">Edit the record</Link></div>
            {loc.error && <div style={{ marginTop: 12 }}><ErrorNote error={loc.error} /></div>}
          </Panel>
          {r && <>
            <Panel title="Why the model output this probability">
              <div className="row spread"><div>Output <ClassPill c={r.predicted_class} /> · P(target = 1) = {pct(r.probability_class1)} · base value {pct(r.base_value)}</div><span className="pill neutral">{r.method === "kernel_shap" ? "KernelSHAP" : "substitution fallback"}</span></div>
              <DecisionStrip p1={r.probability_class1} />
              {r.warning && <div className="note warn" style={{ marginBottom: 10 }}>{r.warning}</div>}
              <ContribChart items={r.contributions} />
              <p className="muted small">Teal bars raise the probability of target = 1, amber bars lower it, relative to a background of {r.background_rows} training records whose average output is {pct(r.base_value)}. Check: base + contributions = {f3(r.base_value + r.sum_of_contributions)} versus the model’s {f3(r.prediction)}{r.additivity_error != null && ` (difference ${r.additivity_error.toExponential(1)})`}.</p>
            </Panel>
            <Panel title="Reading this correctly"><p className="small">{r.reading_guide}</p></Panel></>}
          {!r && !loc.loading && <div className="empty"><h3>No explanation yet</h3>Run the explanation to see per-feature contributions for this record.</div>}
        </div>
      )}
      {tab === "concepts" && (
        <div className="grid g2">{g.data?.concepts.map((c: any) => <Panel key={c.term} title={c.term}><p className="small" style={{ margin: 0 }}>{c.text}</p></Panel>) ?? <Loading />}</div>
      )}
    </>
  );
}
